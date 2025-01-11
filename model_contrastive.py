#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多模态 + 知识图谱 + TransE-like (head+rel ~ tail)
-------------------------------------------------
可调节：batch_size, max_epochs, lr, emb_dim, neg_samples, alpha_mm, freeze_text/image等
包括更多metrics，方便查看结果。
"""

import os
import sys
import csv
import uuid
import random
import argparse
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as T
from PIL import Image
import pandas as pd

import pytorch_lightning as pl
from lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

# TorchMetrics
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torchmetrics.retrieval import RetrievalMRR, RetrievalRecall

# Transformers
from transformers import (
    CLIPModel,
    CLIPTokenizer,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    ViTMAEModel
)

# LR Scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


##############################################################################
# Default Hyperparams (可以在命令行修改)
##############################################################################
DEFAULT_MAX_LENGTH   = 128
DEFAULT_BATCH_SIZE   = 8
DEFAULT_MAX_EPOCHS   = 20
DEFAULT_LR           = 1e-4
DEFAULT_EMB_DIM      = 256
DEFAULT_NEG_SAMPLES  = 5
DEFAULT_ALPHA_MM     = 1.0

FREEZE_TEXT  = True
FREEZE_IMAGE = True

LR_SCHEDULER_CHOICES = ["none", "cosine", "plateau"]

##############################################################################
# Data Paths
##############################################################################
####################################################
# Local Model Paths (本地预训练权重)
####################################################
XLMR_MODEL_PATH     = "/home/shengguang/.cache/huggingface/hub/models--FacebookAI--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
VITMAE_MODEL_PATH   = "/home/shengguang/.cache/huggingface/hub/models--facebook--vit-mae-base/snapshots/25b184bea5538bf5c4c852c79d221195fdd2778d"
CLIP_MODEL_PATH     = "/home/shengguang/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

####################################################
# Data Paths
####################################################
CSV_FILE       = "/home/shengguang/PycharmProjects/movielens_recommendation/FishGraph/FishGraph/annotation.csv"
IMAGE_ROOT_DIR = "/home/shengguang/PycharmProjects/movielens_recommendation/FishGraph/FishGraph/标注后的图像数据改名后的"


##############################################################################
# 1) 动态读取CSV => 多列 => 生成(病名, 关系, 对象)
##############################################################################
REL_COLS = {
    "感染对象":  "感染",
    "症状":     "症状",
    "流行季节": "流行季节",
    "流行地区": "流行地区",
    # 可再添加
    "地区":     "地区",
    "季节":     "季节",
    "温度":     "温度",
    "时间":     "时间",
}

def load_dynamic_csv(csv_file: str, image_root: str):
    """
    返回 list of dict:
    {
      "disease": str(病名),
      "relation": str(REL_COLS对应的值),
      "object": str(对象值),
      "image_path": str(完整路径),
      "text": str(文本)
    }
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    df = pd.read_csv(csv_file, encoding="utf-8", dtype=str).fillna("")
    # 必需列
    must_cols = ["病名", "image_path", "text"]
    for c in must_cols:
        if c not in df.columns:
            raise ValueError(f"Missing col={c} in CSV")

    df["image_path"] = df["image_path"].apply(lambda x: str(x).replace("\\","/"))

    out_data = []
    for _, row in df.iterrows():
        disease_str = row["病名"].strip()
        if not disease_str:
            continue
        img_path = row["image_path"].strip()
        full_img_path = os.path.join(image_root, img_path)
        text_str = row["text"].strip()

        for colName, relName in REL_COLS.items():
            if colName in df.columns:
                val = str(row[colName]).strip()
                if val:
                    out_data.append({
                        "disease":   disease_str,
                        "relation":  relName,
                        "object":    val,
                        "image_path": full_img_path,
                        "text":      text_str
                    })
    return out_data


##############################################################################
# 2) Dataset => 负采样 => PIL => Tensor
##############################################################################
class MultiModalTripleDataset(Dataset):
    def __init__(self, records, ent2id, rel2id, neg_samples=5):
        super().__init__()
        self.records    = records
        self.ent2id     = ent2id
        self.rel2id     = rel2id
        self.neg_samples= neg_samples
        self.all_entities= list(ent2id.keys())

        self.img_transform= T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            # T.Normalize(...) => 可选
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        disease_str = rec["disease"]
        rel_str     = rec["relation"]
        obj_str     = rec["object"]
        img_path    = rec["image_path"]
        text_str    = rec["text"]

        head_id = self.ent2id.get(disease_str, 0)
        rel_id  = self.rel2id.get(rel_str, 0)
        tail_id = self.ent2id.get(obj_str, 0)

        if not os.path.exists(img_path):
            pil_img= Image.new("RGB",(224,224),color="black")
        else:
            pil_img= Image.open(img_path).convert("RGB")

        img_tsr= self.img_transform(pil_img)

        # neg
        neg_ids= []
        for _ in range(self.neg_samples):
            cand = random.choice(self.all_entities)
            while cand == obj_str:
                cand = random.choice(self.all_entities)
            neg_ids.append(self.ent2id[cand])
        neg_tails= torch.tensor(neg_ids, dtype=torch.long)

        return {
            "head_id":  head_id,
            "rel_id":   rel_id,
            "tail_id":  tail_id,
            "image":    img_tsr,
            "text":     text_str,
            "neg_tails":neg_tails
        }


##############################################################################
# 3) CBAM
##############################################################################
class CBAM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim//4),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim)
        )
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        a= self.fc(x)
        a= self.sigmoid(a)
        return x*a


##############################################################################
# 4) TransE + MultiModal
##############################################################################
class KGMulModalLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr=DEFAULT_LR,
        ent_total=100,
        rel_total=5,
        emb_dim=DEFAULT_EMB_DIM,
        freeze_text=FREEZE_TEXT,
        freeze_image=FREEZE_IMAGE,
        neg_samples=DEFAULT_NEG_SAMPLES,
        alpha_mm=DEFAULT_ALPHA_MM,
        lr_scheduler="none",
        xlm_path="",
        clip_path="",
        vitmae_path=""
    ):
        super().__init__()
        self.save_hyperparameters()
        # 1+neg => classes
        self.num_classes= neg_samples+1

        # entity + relation embeddings
        self.ent_embed= nn.Embedding(ent_total, emb_dim)
        self.rel_embed= nn.Embedding(rel_total, emb_dim)

        # XLM-R
        self.xlm_tokenizer= XLMRobertaTokenizer.from_pretrained(xlm_path)
        self.xlm_model= XLMRobertaModel.from_pretrained(xlm_path)
        self.xlm_hidden= self.xlm_model.config.hidden_size

        # CLIP
        self.clip_model= CLIPModel.from_pretrained(clip_path)
        self.clip_proj_dim= self.clip_model.config.projection_dim

        # ViTMAE
        self.vitmae_model= ViTMAEModel.from_pretrained(vitmae_path)
        self.vitmae_hidden= self.vitmae_model.config.hidden_size

        if freeze_text:
            for p in self.xlm_model.parameters():
                p.requires_grad= False
            for p in self.clip_model.text_model.parameters():
                p.requires_grad= False
        if freeze_image:
            for p in self.clip_model.vision_model.parameters():
                p.requires_grad= False
            for p in self.vitmae_model.parameters():
                p.requires_grad= False

        # sub-fusions
        text_in= self.xlm_hidden + self.clip_proj_dim
        self.text_cbam= CBAM(text_in)
        self.text_proj= nn.Linear(text_in, emb_dim)

        img_in= self.vitmae_hidden + self.clip_proj_dim
        self.img_cbam= CBAM(img_in)
        self.img_proj= nn.Linear(img_in, emb_dim)

        self.fusion= CBAM(2*emb_dim)
        self.fusion_proj= nn.Linear(2*emb_dim, emb_dim)

        self.ce_loss= nn.CrossEntropyLoss()

        # metrics
        self.train_acc   = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_prec  = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.train_recall= Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.train_f1    = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

        self.val_acc     = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_prec    = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_recall  = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1      = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

        self.train_mrr= RetrievalMRR()
        self.train_r1 = RetrievalRecall(top_k=1)
        self.val_mrr  = RetrievalMRR()
        self.val_r1   = RetrievalRecall(top_k=1)

    ########################################################################
    # encode text
    ########################################################################
    def encode_text_xlm(self, text_list, device):
        enc= self.xlm_tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to(device)
        out= self.xlm_model(**enc)
        emb= out.last_hidden_state[:,0,:]
        return emb

    def encode_text_clip(self, text_list, device):
        tk= CLIPTokenizer.from_pretrained(self.hparams.clip_path)(
            text_list, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        emb= self.clip_model.get_text_features(**tk)
        return emb

    def encode_text_mm(self, text_list, device):
        x1= self.encode_text_xlm(text_list, device=device)
        x2= self.encode_text_clip(text_list, device=device)
        cat= torch.cat([x1,x2],dim=1)
        cat= self.text_cbam(cat)
        out= self.text_proj(cat)
        return out

    ########################################################################
    # encode image
    ########################################################################
    def encode_image_mm(self, images_tensor, device):
        images_tensor= images_tensor.to(device)
        vout= self.vitmae_model(pixel_values=images_tensor)
        v_emb= vout.last_hidden_state.mean(dim=1)
        clip_emb= self.clip_model.get_image_features(pixel_values=images_tensor)
        cat= torch.cat([v_emb, clip_emb],dim=1)
        cat= self.img_cbam(cat)
        out= self.img_proj(cat)
        return out

    ########################################################################
    # get entity / relation
    ########################################################################
    def get_ent_embed(self, ent_ids):
        emb= self.ent_embed(ent_ids)
        emb= emb/(emb.norm(dim=1,keepdim=True)+1e-9)
        return emb

    def get_rel_embed(self, rel_ids):
        emb= self.rel_embed(rel_ids)
        emb= emb/(emb.norm(dim=1,keepdim=True)+1e-9)
        return emb

    ########################################################################
    # head = base + alpha_mm * mm
    ########################################################################
    def get_head_embed(self, head_ids, mm_emb):
        base= self.get_ent_embed(head_ids)
        out= base + self.hparams.alpha_mm* mm_emb
        out= out/(out.norm(dim=1,keepdim=True)+1e-9)
        return out

    ########################################################################
    # triple score => L2
    ########################################################################
    def triple_score(self, h, r, t):
        diff= (h+r)-t
        sc= diff.norm(p=2,dim=1)
        return sc

    ########################################################################
    # forward => [B, 1+K]
    ########################################################################
    def forward(self, heads, rels, tails, text_list, images_tsr, neg_tails):
        device= heads.device
        txt_e= self.encode_text_mm(text_list, device=device)
        img_e= self.encode_image_mm(images_tsr, device=device)

        cat= torch.cat([txt_e, img_e],dim=1)
        cat= self.fusion(cat)
        mm_fused= self.fusion_proj(cat)

        head_e= self.get_head_embed(heads, mm_fused)
        rel_e = self.get_rel_embed(rels)
        tail_e= self.get_ent_embed(tails)
        pos_sc= self.triple_score(head_e, rel_e, tail_e)  # [B]

        B,K= neg_tails.size(0), neg_tails.size(1)
        negf= neg_tails.view(-1)
        neg_emb= self.get_ent_embed(negf)
        head_exp= head_e.unsqueeze(1).expand(B,K,head_e.size(1)).reshape(B*K,head_e.size(1))
        rel_exp = rel_e.unsqueeze(1).expand(B,K,rel_e.size(1)).reshape(B*K,rel_e.size(1))
        neg_sc= self.triple_score(head_exp, rel_exp, neg_emb).view(B,K)

        all_sc= torch.cat([pos_sc.unsqueeze(1), neg_sc],dim=1)
        return all_sc

    ########################################################################
    # training step
    ########################################################################
    def training_step(self, batch, batch_idx):
        device= self.device
        heads= batch["head_id"].to(device)
        rels=  batch["rel_id"].to(device)
        tails= batch["tail_id"].to(device)
        text_list= batch["text"]
        imgs= batch["image"].to(device)
        negs= batch["neg_tails"].to(device)

        all_sc= self(heads, rels, tails, text_list, imgs, negs)
        label= torch.zeros_like(heads, dtype=torch.long, device=device)
        loss= self.ce_loss(all_sc, label)

        # classification
        preds= all_sc.argmin(dim=1)  # 0 => pos
        self.train_acc.update(preds, label)
        self.train_prec.update(preds, label)
        self.train_recall.update(preds, label)
        self.train_f1.update(preds, label)

        # retrieval
        B,K= negs.size()
        indexes= torch.arange(B, device=device).unsqueeze(1).expand(B,1+K).reshape(-1)
        lab_= torch.zeros((B,1+K), dtype=torch.long, device=device)
        lab_[:,0]=1
        sc_flat= all_sc.reshape(-1)
        lab_flat= lab_.reshape(-1)
        self.train_mrr.update(sc_flat, lab_flat, indexes)
        self.train_r1.update(sc_flat, lab_flat, indexes)

        self.log("train_loss", loss, on_step=True,on_epoch=True,batch_size=B)
        return loss

    def on_train_epoch_end(self):
        # classification
        acc= self.train_acc.compute()
        prec= self.train_prec.compute()
        rec= self.train_recall.compute()
        f1= self.train_f1.compute()
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_prec", prec)
        self.log("train_recall", rec)
        self.log("train_f1", f1)
        self.train_acc.reset()
        self.train_prec.reset()
        self.train_recall.reset()
        self.train_f1.reset()

        # retrieval
        mrr= self.train_mrr.compute()
        r1= self.train_r1.compute()
        self.log("train_mrr", mrr, prog_bar=True)
        self.log("train_r1", r1)
        self.train_mrr.reset()
        self.train_r1.reset()

    ########################################################################
    # validation step
    ########################################################################
    def validation_step(self, batch, batch_idx):
        device= self.device
        B= batch["head_id"].size(0)
        heads= batch["head_id"].to(device)
        rels=  batch["rel_id"].to(device)
        tails= batch["tail_id"].to(device)
        imgs= batch["image"].to(device)
        text_list= batch["text"]
        negs= batch["neg_tails"].to(device)

        all_sc= self(heads, rels, tails, text_list, imgs, negs)
        label= torch.zeros_like(heads,dtype=torch.long,device=device)
        loss= self.ce_loss(all_sc, label)

        preds= all_sc.argmin(dim=1)
        self.val_acc.update(preds, label)
        self.val_prec.update(preds, label)
        self.val_recall.update(preds, label)
        self.val_f1.update(preds, label)

        # retrieval
        K= negs.size(1)
        indexes= torch.arange(B, device=device).unsqueeze(1).expand(B,1+K).reshape(-1)
        lab_= torch.zeros((B,1+K), dtype=torch.long, device=device)
        lab_[:,0]=1
        sc_flat= all_sc.view(-1)
        lab_flat= lab_.view(-1)
        self.val_mrr.update(sc_flat, lab_flat, indexes)
        self.val_r1.update(sc_flat, lab_flat, indexes)

        self.log("val_loss", loss, prog_bar=True,batch_size=B)

        if batch_idx==0:
            sample_num= min(3,B)
            logs=[]
            for i in range(sample_num):
                logs.append({
                    "disease_text": text_list[i],
                    "score_pos": float(all_sc[i,0].item()),
                    "score_neg": [float(x.item()) for x in all_sc[i,1:]],
                    "argmin": int(preds[i].item()),
                    "loss": float(loss.item())
                })
            self.logger.experiment.log({"val_samples": logs})

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # classification
        acc= self.val_acc.compute()
        prec= self.val_prec.compute()
        rec= self.val_recall.compute()
        f1=  self.val_f1.compute()
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_prec", prec)
        self.log("val_recall", rec)
        self.log("val_f1", f1)
        self.val_acc.reset()
        self.val_prec.reset()
        self.val_recall.reset()
        self.val_f1.reset()

        # retrieval
        mrr= self.val_mrr.compute()
        r1= self.val_r1.compute()
        self.log("val_mrr", mrr, prog_bar=True)
        self.log("val_r1", r1)
        self.val_mrr.reset()
        self.val_r1.reset()

    ########################################################################
    # configure_optimizers
    ########################################################################
    def configure_optimizers(self):
        opt= torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch_type= self.hparams.lr_scheduler
        if sch_type=="cosine":
            sched= CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
            return [opt],[sched]
        elif sch_type=="plateau":
            sched= ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)
            return {
                "optimizer": opt,
                "lr_scheduler":{
                    "scheduler": sched,
                    "monitor": "val_loss",
                    "interval":"epoch",
                    "frequency":1
                }
            }
        else:
            return opt


##############################################################################
# main
##############################################################################
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--csv_file",     type=str, default=CSV_FILE)
    parser.add_argument("--image_dir",    type=str, default=IMAGE_ROOT_DIR)
    parser.add_argument("--run_id",       type=str, default=None)
    parser.add_argument("--project_name", type=str, default="kge_mm_transE_more_metrics")

    parser.add_argument("--batch_size",   type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_epochs",   type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--lr",           type=float, default=DEFAULT_LR)
    parser.add_argument("--freeze_text",  action="store_true", default=FREEZE_TEXT)
    parser.add_argument("--freeze_image", action="store_true", default=FREEZE_IMAGE)
    parser.add_argument("--neg_samples",  type=int, default=DEFAULT_NEG_SAMPLES)
    parser.add_argument("--emb_dim",      type=int, default=DEFAULT_EMB_DIM)
    parser.add_argument("--alpha_mm",     type=float, default=DEFAULT_ALPHA_MM)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=LR_SCHEDULER_CHOICES)

    # huggingface local paths
    parser.add_argument("--xlm_path",     type=str, default=XLMR_MODEL_PATH)
    parser.add_argument("--clip_path",    type=str, default=CLIP_MODEL_PATH)
    parser.add_argument("--vitmae_path",  type=str, default=VITMAE_MODEL_PATH)

    args= parser.parse_args()
    if args.run_id is None:
        args.run_id= str(uuid.uuid4())[:8]

    seed_everything(42)

    # 1) load triple from CSV
    triple_data= load_dynamic_csv(args.csv_file, args.image_dir)
    if not triple_data:
        print("[Error] CSV empty or no triple!")
        sys.exit(1)
    random.shuffle(triple_data)

    # 2) build ent2id, rel2id
    all_d= {r["disease"] for r in triple_data}
    all_o= {r["object"]  for r in triple_data}
    all_e= sorted(list(all_d.union(all_o)))
    ent2id= { e:i for i,e in enumerate(all_e) }

    all_r= sorted({ r["relation"] for r in triple_data })
    rel2id= { rr:i for i,rr in enumerate(all_r) }

    # 3) split
    sp= int(0.8* len(triple_data))
    train_recs= triple_data[:sp]
    val_recs  = triple_data[sp:]

    train_ds= MultiModalTripleDataset(train_recs, ent2id, rel2id, neg_samples=args.neg_samples)
    val_ds  = MultiModalTripleDataset(val_recs,   ent2id, rel2id, neg_samples=args.neg_samples)

    from torch.utils.data import DataLoader, default_collate
    train_loader= DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,num_workers=2, collate_fn=default_collate)
    val_loader=   DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,num_workers=2, collate_fn=default_collate)

    # 4) model
    model= KGMulModalLightningModule(
        lr=args.lr,
        ent_total=len(ent2id),
        rel_total=len(rel2id),
        emb_dim=args.emb_dim,
        freeze_text=args.freeze_text,
        freeze_image=args.freeze_image,
        neg_samples=args.neg_samples,
        alpha_mm=args.alpha_mm,
        lr_scheduler=args.lr_scheduler,
        xlm_path=args.xlm_path,
        clip_path=args.clip_path,
        vitmae_path=args.vitmae_path
    )

    # 5) trainer
    run_dir= f"runs/run_{args.run_id}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir= os.path.join(run_dir,"ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_logger= WandbLogger(project=args.project_name, name=f"mmKGE_{args.run_id}", save_dir=run_dir)
    ckpt_callback= ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        mode="min",
        filename="best",
        save_top_k=1,
        save_last=True
    )
    lr_monitor= LearningRateMonitor(logging_interval='step')

    trainer= pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=run_dir,
        logger=wandb_logger,
        callbacks=[ckpt_callback, lr_monitor],
        precision=16,  # or "16-mixed"
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"[Done] best ckpt= {ckpt_dir}")


if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()
