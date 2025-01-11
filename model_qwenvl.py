#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多模态 + 知识图谱 + RotatE + Qwen2-VL
-------------------------------------------------
改进点：
1. 使用Qwen2-VL作为多模态编码器（可做文本+图像多模态特征抽取）。
2. 替换TransE为RotatE，用来建模实体-关系三元组。
3. 可选地实现“硬负采样”策略。
4. 用CrossAttention(如MultiheadAttention)替代CBAM。
5. 将Head embedding与多模态embedding融合改为可学习的权重(投影等)。
6. 修复了两个常见错误：
   - “embedding dimension mismatch” => 通过投影 Qwen2VL输出 => 512，再CrossAttention => emb_dim
   - “IndexError: list index out of range” => 通过补齐neg_tail_texts外层/内层长度
   - “TypeError: can only concatenate tuple (not 'list') ...” => 在拼接前转list
"""

import os
import sys
import uuid
import random
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as T
from PIL import Image
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

# TorchMetrics
from torchmetrics.retrieval import RetrievalMRR, RetrievalRecall

# Transformers
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
# LR 调度器
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

##############################################################################
# Default Hyperparams
##############################################################################
DEFAULT_MAX_LENGTH   = 128
DEFAULT_BATCH_SIZE   = 32
DEFAULT_MAX_EPOCHS   = 100
DEFAULT_LR           = 1e-4
DEFAULT_EMB_DIM      = 256
DEFAULT_NEG_SAMPLES  = 2
DEFAULT_ALPHA_MM     = 1.0

FREEZE_TEXT  = True
FREEZE_IMAGE = True

LR_SCHEDULER_CHOICES = ["none", "cosine", "plateau"]
DEFAULT_DEBUG_FLAG   = False

##############################################################################
# Paths
##############################################################################
QWEN2_VL_MODEL_PATH = (
    "/home/shengguang/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/47592516d3e709cd9c194715bc76902241c5edea"
)
CSV_FILE       = "/home/shengguang/PycharmProjects/movielens_recommendation/FishGraph/FishGraph/annotation.csv"
IMAGE_ROOT_DIR = "/home/shengguang/PycharmProjects/movielens_recommendation/FishGraph/FishGraph/标注后的图像数据改名后的"

##############################################################################
# 1) 读取CSV => (病名, 关系, 对象)
##############################################################################
REL_COLS = {
    "感染对象":  "感染",
    "症状":     "症状",
    "流行季节": "流行季节",
    "流行地区": "流行地区",
    "地区":     "地区",
    "季节":     "季节",
    "温度":     "温度",
    "时间":     "时间",
}

def load_dynamic_csv(csv_file: str, image_root: str):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    df = pd.read_csv(csv_file, encoding="utf-8", dtype=str).fillna("")
    must_cols = ["病名", "image_path", "text"]
    for c in must_cols:
        if c not in df.columns:
            raise ValueError(f"Missing col={c} in CSV")

    df["image_path"] = df["image_path"].apply(lambda x: str(x).replace("\\", "/"))
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
# 2) Dataset => 保证 neg_tail_texts 与 neg_samples匹配
##############################################################################
class MultiModalTripleDataset(Dataset):
    def __init__(
        self,
        records,
        ent2id,
        rel2id,
        neg_samples=2,
        obj_name_map=None,
        hard_negatives: Dict[str, List[str]] = None
    ):
        super().__init__()
        self.records       = records
        self.ent2id        = ent2id
        self.rel2id        = rel2id
        self.neg_samples   = neg_samples
        self.all_entities  = list(ent2id.keys())
        self.obj_name_map  = obj_name_map or {}
        self.hard_negatives= hard_negatives or {}

        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        disease_str   = rec["disease"]
        rel_str       = rec["relation"]
        obj_str       = rec["object"]
        img_path      = rec["image_path"]
        head_text_str = rec["text"]
        tail_text_str = self.obj_name_map.get(obj_str, obj_str)

        head_id = self.ent2id.get(disease_str, 0)
        rel_id  = self.rel2id.get(rel_str, 0)
        tail_id = self.ent2id.get(obj_str, 0)

        if not os.path.exists(img_path):
            pil_img = Image.new("RGB",(224,224), color="black")
        else:
            pil_img = Image.open(img_path).convert("RGB")
        img_tsr = self.img_transform(pil_img)

        # 负采样 => 返回 neg_samples 个
        neg_ids   = []
        neg_texts = []
        for _ in range(self.neg_samples):
            hard_neg_list = self.hard_negatives.get(obj_str)
            if hard_neg_list:
                cand = random.choice(hard_neg_list)
                while (cand == obj_str) or (cand not in self.ent2id):
                    cand = random.choice(self.all_entities)
            else:
                cand = random.choice(self.all_entities)
                while cand == obj_str:
                    cand = random.choice(self.all_entities)

            neg_ids.append(self.ent2id[cand])
            neg_texts.append(self.obj_name_map.get(cand, cand))

        neg_tails = torch.tensor(neg_ids, dtype=torch.long)

        return {
            "head_id":       head_id,
            "rel_id":        rel_id,
            "tail_id":       tail_id,
            "image":         img_tsr,
            "head_text":     head_text_str,
            "tail_text":     tail_text_str,
            "neg_tails":     neg_tails,
            "neg_tail_texts": neg_texts  # len=neg_samples
        }

##############################################################################
# 3) CrossAttention => embed_dim=512
##############################################################################
class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(attn_output + x)
        ffn_output = self.ffn(x)
        x = self.layer_norm2(ffn_output + x)
        return x

##############################################################################
# 4) RotatE
##############################################################################
class RotatE(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        assert emb_dim % 2 == 0, "Embedding dimension must be even for RotatE."

    def forward(self, h, r, t):
        h_real, h_imag = torch.chunk(h, 2, dim=-1)
        r_real, r_imag = torch.chunk(r, 2, dim=-1)
        t_real, t_imag = torch.chunk(t, 2, dim=-1)

        t_pred_real = h_real * r_real - h_imag * r_imag
        t_pred_imag = h_real * r_imag + h_imag * r_real

        real_diff = t_pred_real - t_real
        imag_diff = t_pred_imag - t_imag
        score = torch.norm(torch.cat([real_diff, imag_diff], dim=-1), dim=-1)
        return score

##############################################################################
# 5) LightningModule
##############################################################################
class KGMulModalLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        ent_total=100,
        rel_total=5,
        emb_dim=256,
        freeze_text=True,
        freeze_image=True,
        neg_samples=2,
        alpha_mm=1.0,
        margin=1.0,
        lr_scheduler="none",
        debug=False,
        qwen2_vl_model_name=QWEN2_VL_MODEL_PATH,
        use_image_prompt=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.margin = margin  # 新增 margin 超参数
        self.debug= debug
        self.use_image_prompt = use_image_prompt

        # RotatE
        self.ent_embed= nn.Embedding(ent_total, emb_dim)
        self.rel_embed= nn.Embedding(rel_total, emb_dim)
        self.rotate = RotatE(emb_dim=emb_dim)

        # Qwen2-VL
        self.processor = Qwen2VLProcessor.from_pretrained(qwen2_vl_model_name)
        self.qwen2_vl = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen2_vl_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        if freeze_text:
            for param in self.qwen2_vl.parameters():
                param.requires_grad=False

        # 维度投影
        self.qwen_hidden_size = self.qwen2_vl.config.hidden_size  # e.g.1536
        self.proj_dim = 512
        self.qwen_proj = nn.Linear(self.qwen_hidden_size, self.proj_dim)

        self.cross_attention = CrossAttentionModule(embed_dim=self.proj_dim)
        self.fusion_proj = nn.Linear(self.proj_dim, emb_dim)

        # 损失函数改为 Margin Ranking Loss
        self.margin_loss = nn.MarginRankingLoss(margin=self.margin)

        # 仅保留排名指标
        self.train_mrr= RetrievalMRR()
        self.train_r1 = RetrievalRecall(top_k=1)
        self.val_mrr  = RetrievalMRR()
        self.val_r1   = RetrievalRecall(top_k=1)

    # =========== utils ===========
    def get_ent_embed(self, ent_ids):
        emb= self.ent_embed(ent_ids)
        emb= emb/(emb.norm(dim=1, keepdim=True)+1e-9)
        return emb

    def get_rel_embed(self, rel_ids):
        emb= self.rel_embed(rel_ids)
        emb= emb/(emb.norm(dim=1, keepdim=True)+1e-9)
        return emb

    def triple_score(self, h, r, t):
        return self.rotate(h, r, t)

    # =========== encode multimodal ===========
    def encode_multimodal(self, text_list, image_tensor=None):
        if self.use_image_prompt and (image_tensor is not None):
            text_list = [f"<|image|> {t}" for t in text_list]
            pil_img = T.ToPILImage()(image_tensor)
            images_for_proc = [pil_img]
        else:
            images_for_proc = None

        inputs = self.processor(
            text=text_list,
            images=images_for_proc,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            out= self.qwen2_vl(**inputs, output_hidden_states=True)
            hidden_states= out.hidden_states[-1]
            cls_emb= hidden_states[:,0,:]

        cls_proj= self.qwen_proj(cls_emb)       # [B, 512]
        x= cls_proj.unsqueeze(1)               # [B,1,512]
        x= self.cross_attention(x)             # [B,1,512]
        x= self.fusion_proj(x).squeeze(1)      # [B, emb_dim=256]
        return x

    # =========== forward ===========
    def forward(self,
                heads, rels, tails,
                head_text_list, head_imgs,
                tail_text_list,
                neg_tail_texts, neg_tails):
        B= heads.size(0)
        K= neg_tails.size(1)

        # head mm
        head_mm_list= []
        for i in range(B):
            txt_i= [head_text_list[i]]
            img_i= head_imgs[i]
            mm_i= self.encode_multimodal(txt_i, img_i)
            head_mm_list.append(mm_i)
        head_mm= torch.cat(head_mm_list, dim=0)  # [B, emb_dim]

        head_e_base= self.get_ent_embed(heads)   # [B, emb_dim]
        head_e= head_e_base + self.hparams.alpha_mm * head_mm
        head_e= head_e / (head_e.norm(dim=1, keepdim=True) + 1e-9)

        # rel
        rel_e= self.get_rel_embed(rels)         # [B, emb_dim]

        # tail mm
        tail_mm_list= []
        for i in range(B):
            txt_i= [tail_text_list[i]]
            mm_i= self.encode_multimodal(txt_i, None)
            tail_mm_list.append(mm_i)
        tail_mm= torch.cat(tail_mm_list, dim=0)  # [B, emb_dim]
        tail_e_base= self.get_ent_embed(tails)   # [B, emb_dim]
        tail_e= tail_e_base + self.hparams.alpha_mm * tail_mm
        tail_e= tail_e / (tail_e.norm(dim=1, keepdim=True) + 1e-9)

        pos_sc= self.triple_score(head_e, rel_e, tail_e)  # [B]

        # negative
        neg_ids_flat= neg_tails.reshape(-1)              # [B*K]
        neg_e_base= self.get_ent_embed(neg_ids_flat)     # [B*K, emb_dim]

        # flatten neg_texts
        neg_texts_flat=[]
        for i in range(B):
            # 如果 i >= len(neg_tail_texts)，则补一个空
            if i >= len(neg_tail_texts):
                neg_tail_texts.append(["Unknown"]*K)

            sample_list= neg_tail_texts[i]

            # tuple => 需转成 list
            if isinstance(sample_list, tuple):
                sample_list = list(sample_list)

            # 补齐 or 截断
            if len(sample_list) < K:
                last_txt= sample_list[-1] if sample_list else "Unknown"
                sample_list = sample_list + [last_txt]*(K - len(sample_list))
            elif len(sample_list) > K:
                sample_list= sample_list[:K]

            neg_texts_flat.extend(sample_list)

        neg_mm_list= []
        for idx, neg_str in enumerate(neg_texts_flat):
            mm_ij= self.encode_multimodal([neg_str], None)
            neg_mm_list.append(mm_ij)
        neg_mm= torch.cat(neg_mm_list, dim=0)          # [B*K, emb_dim]
        neg_e= neg_e_base + self.hparams.alpha_mm * neg_mm
        neg_e= neg_e / (neg_e.norm(dim=1, keepdim=True) + 1e-9)  # [B*K, emb_dim]

        h_e_exp= head_e.unsqueeze(1).expand(B, K, head_e.size(1)).reshape(B*K, head_e.size(1))  # [B*K, emb_dim]
        r_e_exp= rel_e.unsqueeze(1).expand(B, K, rel_e.size(1)).reshape(B*K, rel_e.size(1))    # [B*K, emb_dim]
        neg_sc= self.triple_score(h_e_exp, r_e_exp, neg_e).view(B, K)                         # [B, K]

        return pos_sc, neg_sc  # [B], [B, K]

    # =========== training_step ===========
    def training_step(self, batch, batch_idx):
        heads= batch["head_id"].to(self.device)
        rels=  batch["rel_id"].to(self.device)
        tails= batch["tail_id"].to(self.device)
        head_text_list= batch["head_text"]
        imgs= batch["image"].to(self.device)
        tail_text_list= batch["tail_text"]
        negs= batch["neg_tails"].to(self.device)
        neg_texts= batch["neg_tail_texts"]

        pos_sc, neg_sc = self(
            heads, rels, tails,
            head_text_list, imgs,
            tail_text_list,
            neg_texts, negs
        )  # pos_sc => [B], neg_sc => [B, K]

        B, K = neg_sc.size()

        # 准备 preds、target 和 indexes
        preds = torch.cat([pos_sc.unsqueeze(1), neg_sc], dim=1).view(-1)  # [B*(K+1)]
        target = torch.cat([torch.ones(B, 1, device=self.device), torch.zeros(B, K, device=self.device)], dim=1).view(-1)  # [B*(K+1)]
        indexes = torch.arange(B, device=self.device).unsqueeze(1).repeat(1, K+1).view(-1)  # [B*(K+1)]

        # 定义 margin loss
        # 假设 score 越小越好（TransE/RotatE）
        # 通过取负值使得更高的分数表示更好的预测
        preds_neg = -preds  # [B*(K+1)]
        loss = self.margin_loss(preds_neg, torch.ones_like(preds_neg, device=self.device), target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=B)

        # 更新排名指标
        self.train_mrr.update(preds_neg, target, indexes)
        self.train_r1.update(preds_neg, target, indexes)

        return loss

    def on_train_epoch_end(self):
        mrr= self.train_mrr.compute()
        r1= self.train_r1.compute()
        self.log("train_mrr", mrr, prog_bar=True)
        self.log("train_r1", r1)
        self.train_mrr.reset()
        self.train_r1.reset()

    def validation_step(self, batch, batch_idx):
        heads= batch["head_id"].to(self.device)
        rels=  batch["rel_id"].to(self.device)
        tails= batch["tail_id"].to(self.device)
        head_text_list= batch["head_text"]
        imgs= batch["image"].to(self.device)
        tail_text_list= batch["tail_text"]
        negs= batch["neg_tails"].to(self.device)
        neg_texts= batch["neg_tail_texts"]

        pos_sc, neg_sc = self(
            heads, rels, tails,
            head_text_list, imgs,
            tail_text_list,
            neg_texts, negs
        )  # pos_sc => [B], neg_sc => [B, K]

        B, K = neg_sc.size()

        # 准备 preds、target 和 indexes
        preds = torch.cat([pos_sc.unsqueeze(1), neg_sc], dim=1).view(-1)  # [B*(K+1)]
        target = torch.cat([torch.ones(B, 1, device=self.device), torch.zeros(B, K, device=self.device)], dim=1).view(-1)  # [B*(K+1)]
        indexes = torch.arange(B, device=self.device).unsqueeze(1).repeat(1, K+1).view(-1)  # [B*(K+1)]

        # 定义 margin loss
        preds_neg = -preds  # [B*(K+1)]
        loss = self.margin_loss(preds_neg, torch.ones_like(preds_neg, device=self.device), target)

        self.log("val_loss", loss, prog_bar=True, batch_size=B)

        # 更新排名指标
        self.val_mrr.update(preds_neg, target, indexes)
        self.val_r1.update(preds_neg, target, indexes)

        if batch_idx == 0 and self.debug:
            # 仅在调试模式下记录一些样本
            sample_num= min(3, heads.size(0))
            for i in range(sample_num):
                print(f"Sample {i}:")
                print(f"Head: {head_text_list[i]}")
                print(f"Relation: {rels[i].item()}")
                print(f"True Tail: {tail_text_list[i]}")
                print(f"Pos Score: {pos_sc[i].item()}")
                print(f"Neg Scores: {neg_sc[i].tolist()}")
                print("-" * 20)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        mrr= self.val_mrr.compute()
        r1= self.val_r1.compute()
        self.log("val_mrr", mrr, prog_bar=True)
        self.log("val_r1", r1)
        self.val_mrr.reset()
        self.val_r1.reset()

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
    parser.add_argument("--project_name", type=str, default="mmKGE_with_Qwen2VL")

    parser.add_argument("--batch_size",   type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_epochs",   type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--lr",           type=float, default=DEFAULT_LR)
    parser.add_argument("--freeze_text",  action="store_true", default=FREEZE_TEXT)
    parser.add_argument("--freeze_image", action="store_true", default=FREEZE_IMAGE)
    parser.add_argument("--neg_samples",  type=int, default=DEFAULT_NEG_SAMPLES)
    parser.add_argument("--emb_dim",      type=int, default=DEFAULT_EMB_DIM)
    parser.add_argument("--alpha_mm",     type=float, default=DEFAULT_ALPHA_MM)
    parser.add_argument("--margin",       type=float, default=1.0, help="Margin for ranking loss")
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=LR_SCHEDULER_CHOICES)
    parser.add_argument("--debug", action="store_true", default=DEFAULT_DEBUG_FLAG)
    parser.add_argument("--qwen2_vl_model_name", type=str, default=QWEN2_VL_MODEL_PATH)

    parser.add_argument("--use_image_prompt", action="store_true",
                        help="If true, insert <|image|> into text prompt so Qwen2-VL can see the image tokens.")

    args= parser.parse_args()
    if args.run_id is None:
        args.run_id= str(uuid.uuid4())[:8]

    seed_everything(42)

    # 1) load
    triple_data= load_dynamic_csv(args.csv_file, args.image_dir)
    if not triple_data:
        print("[Error] CSV empty or no triple!")
        sys.exit(1)
    random.shuffle(triple_data)

    # 2) ent2id, rel2id
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

    obj_name_map= {}
    hard_negatives= {}

    train_ds= MultiModalTripleDataset(
        train_recs, ent2id, rel2id,
        neg_samples=args.neg_samples,
        obj_name_map=obj_name_map,
        hard_negatives=hard_negatives
    )
    val_ds= MultiModalTripleDataset(
        val_recs, ent2id, rel2id,
        neg_samples=args.neg_samples,
        obj_name_map=obj_name_map,
        hard_negatives=hard_negatives
    )

    train_loader= DataLoader(train_ds, batch_size=args.batch_size,
                             shuffle=True, num_workers=2, persistent_workers=True)
    val_loader=   DataLoader(val_ds,   batch_size=args.batch_size,
                             shuffle=False, num_workers=2, persistent_workers=True)

    model= KGMulModalLightningModule(
        lr=args.lr,
        ent_total=len(ent2id),
        rel_total=len(rel2id),
        emb_dim=args.emb_dim,
        freeze_text=args.freeze_text,
        freeze_image=args.freeze_image,
        neg_samples=args.neg_samples,
        alpha_mm=args.alpha_mm,
        margin=args.margin,
        lr_scheduler=args.lr_scheduler,
        debug=args.debug,
        qwen2_vl_model_name=args.qwen2_vl_model_name,
        use_image_prompt=args.use_image_prompt
    )

    run_dir= f"runs/run_{args.run_id}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir= os.path.join(run_dir,"ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_logger= WandbLogger(
        project=args.project_name,
        name=f"mmKGE_Qwen2VL_{args.run_id}",
        save_dir=run_dir
    )
    ckpt_callback= ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_mrr",
        mode="max",
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
        precision=16,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)

    print(f"[Done] best ckpt= {ckpt_dir}")


if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()
