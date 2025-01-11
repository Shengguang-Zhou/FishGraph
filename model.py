#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import uuid
import random
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as T
from PIL import Image
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

# 使用 torchmetrics 进行多分类评估
from torchmetrics.classification import Accuracy

# ========== transformers相关 ============
from transformers import (
    CLIPModel,
    CLIPTokenizer,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    ViTMAEModel
)

####################################################
# 你的关系类别(可选)
####################################################
RELATIONS = ["感染", "症状", "流行地区", "流行季节"]
LABEL2ID = {rel: i for i, rel in enumerate(RELATIONS)}

####################################################
# Default Hyperparameters
####################################################
DEFAULT_MAX_LENGTH   = 128
DEFAULT_BATCH_SIZE   = 8
DEFAULT_MAX_EPOCHS   = 100
DEFAULT_LR           = 1e-5

FREEZE_TEXT  = True
FREEZE_IMAGE = True
USE_TEXT     = True
USE_IMAGE    = True

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

####################################################
# Step 1: read CSV & parse partial triplets
####################################################
def load_triplets(csv_file: str):
    """
    读取annotation.csv，返回list[{...}]结构:
      {
        "image_path": ...
        "text": ...
        "disease": ...
        "relation": ...
        "object": ...
      }
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file, encoding="utf-8")
    required_cols = ["id", "image_path", "text", "病名", "关系"]
    for c in required_cols:
        if c not in df.columns:
            print(f"[Error] Missing column {c} in CSV!")
            return []

    triplets = []
    for _, row in df.iterrows():
        disease_name = str(row.get("病名", ""))
        relation     = str(row.get("关系", ""))
        obj = row.get("感染对象", None)   # 例如"感染对象"列
        if pd.isna(obj) or str(obj).strip() == "":
            continue

        rec = {
            "image_path": os.path.join(IMAGE_ROOT_DIR, str(row["image_path"]).replace("\\", "/")),
            "text": str(row["text"]),
            "disease": disease_name,
            "relation": relation,
            "object": str(obj).strip()
        }
        triplets.append(rec)
    return triplets

def find_image_file(root_dir, partial_path):
    # 如果 partial_path无后缀,可尝试 webp/jpg... 这里省略
    return ""

####################################################
# Step 2: Dataset
####################################################
class MultiModalDataset(Dataset):
    """
    给定三元组 (disease, relation, object) + text + image
    这里我们做 "给定(disease+relation+文本+图)，输出object" => 多分类
    因此 object要离散ID
    """

    def __init__(self, records, object2id, transform=None):
        self.records   = records
        self.object2id = object2id
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        disease_text = f"{rec['disease']} {rec['relation']} {rec['text']}"

        img_path = rec["image_path"]
        if not os.path.exists(img_path):
            image = Image.new("RGB", (224,224), color="black")
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            # 这里先不transform，因为后面模型里会自己transform
            # 也可以选择在这里transform
            pass

        obj_text = rec["object"]
        obj_id = self.object2id[obj_text]  # 作为 label

        return {
            "disease_text": disease_text,
            "image": image,
            "obj_id": obj_id
        }

####################################################
# Step 3: CBAM (简化)
####################################################
class CBAM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim//4),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.fc(x)
        attn = self.sigmoid(attn)
        return x * attn

####################################################
# Step 4: XLM+CLIP + ViTMAE+CLIP => CBAM => multi-class
####################################################
class XLMClipViTMAECBAMModel(pl.LightningModule):
    def __init__(
        self,
        xlm_path,
        vitmae_path,
        clip_path,
        lr=DEFAULT_LR,
        freeze_text=FREEZE_TEXT,
        freeze_image=FREEZE_IMAGE,
        num_objects=100  # 所有 object 的总数
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_objects = num_objects

        # 1) XLM-R
        self.xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_path)
        self.xlm_model     = XLMRobertaModel.from_pretrained(xlm_path)
        self.xlm_hidden_size = self.xlm_model.config.hidden_size

        # 2) CLIP
        self.clip_model = CLIPModel.from_pretrained(clip_path)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
        self.clip_proj_dim = self.clip_model.config.projection_dim

        # 3) ViTMAE
        self.vitmae_model = ViTMAEModel.from_pretrained(vitmae_path)
        self.vitmae_hidden_size = self.vitmae_model.config.hidden_size

        # freeze
        if freeze_text:
            for p in self.xlm_model.parameters():
                p.requires_grad = False
            for p in self.clip_model.text_model.parameters():
                p.requires_grad = False
        if freeze_image:
            for p in self.clip_model.vision_model.parameters():
                p.requires_grad = False
            for p in self.vitmae_model.parameters():
                p.requires_grad = False

        # CBAM
        self.text_in_dim = self.xlm_hidden_size + self.clip_proj_dim
        self.text_cbam   = CBAM(self.text_in_dim)
        self.text_proj   = nn.Linear(self.text_in_dim, self.clip_proj_dim)

        self.image_in_dim= self.vitmae_hidden_size + self.clip_proj_dim
        self.image_cbam  = CBAM(self.image_in_dim)
        self.image_proj  = nn.Linear(self.image_in_dim, self.clip_proj_dim)

        self.final_in_dim= self.clip_proj_dim * 2
        self.final_cbam  = CBAM(self.final_in_dim)
        self.final_proj  = nn.Linear(self.final_in_dim, self.clip_proj_dim)

        # 最后再接一个分类头 => [proj_dim] => [num_objects]
        self.classifier  = nn.Linear(self.clip_proj_dim, self.num_objects)

        self.loss_fn = nn.CrossEntropyLoss()

        # metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_objects)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_objects)

        # transform
        self.img_transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.48145466,0.4578275,0.40821073),
                        (0.26862954,0.26130258,0.27577711))
        ])

    # --- Encode text: XLM + CLIP => CBAM
    def encode_text_xlm(self, text_list, device):
        enc = self.xlm_tokenizer(
            text_list,
            return_tensors="pt",
            padding=True, truncation=True
        ).to(device)
        out = self.xlm_model(**enc)
        # [B, seq_len, hidden], take CLS
        xlm_emb= out.last_hidden_state[:,0,:]
        return xlm_emb

    def encode_text_clip(self, text_list, device):
        enc = self.clip_tokenizer(
            text_list,
            return_tensors="pt",
            padding=True, truncation=True
        ).to(device)
        txt_emb= self.clip_model.get_text_features(**enc)
        return txt_emb

    def encode_text(self, text_list, device):
        xlm_emb  = self.encode_text_xlm(text_list, device=device)
        clip_emb = self.encode_text_clip(text_list, device=device)
        cat = torch.cat([xlm_emb, clip_emb], dim=1) # [B, xlm_hidden+clip_proj]
        out = self.text_cbam(cat)
        out = self.text_proj(out) # => [B, clip_proj_dim]
        return out

    # --- Encode image: ViTMAE + CLIP => CBAM
    def encode_image_vitmae(self, images, device):
        # images: list of PIL
        ts = []
        for img in images:
            t = self.img_transform(img)
            ts.append(t.unsqueeze(0))
        pixel_values= torch.cat(ts, dim=0).to(device)
        out= self.vitmae_model(pixel_values=pixel_values)
        # last_hidden_state => [B, seq_len, hidden]
        vit_emb= out.last_hidden_state.mean(dim=1)
        return vit_emb

    def encode_image_clip(self, images, device):
        ts=[]
        for img in images:
            t=self.img_transform(img)
            ts.append(t.unsqueeze(0))
        pixel_values=torch.cat(ts,dim=0).to(device)
        clip_emb= self.clip_model.get_image_features(pixel_values=pixel_values)
        return clip_emb

    def encode_image(self, images, device):
        vit_emb = self.encode_image_vitmae(images, device=device)
        clip_emb= self.encode_image_clip(images, device=device)
        cat= torch.cat([vit_emb, clip_emb], dim=1)
        out= self.image_cbam(cat)
        out= self.image_proj(out) # => [B, clip_proj_dim]
        return out

    # --- fuse text & image => final cbam => final_emb
    def fuse_text_image(self, text_emb, image_emb):
        cat= torch.cat([text_emb, image_emb], dim=1)
        out= self.final_cbam(cat)
        out= self.final_proj(out)  # => [B,clip_proj_dim]
        return out

    def forward(self, disease_texts, images):
        """
        前向：把 disease+relation+text => text_emb, 把image => image_emb => final => logits
        """
        device = self.device
        text_emb = self.encode_text(disease_texts, device=device)
        image_emb= self.encode_image(images, device=device)
        fused_emb= self.fuse_text_image(text_emb, image_emb)
        logits   = self.classifier(fused_emb)  # => [B, num_objects]
        return logits

    # --- training_step
    def training_step(self, batch, batch_idx):
        device = self.device
        disease_texts= [b["disease_text"] for b in batch]
        images       = [b["image"]        for b in batch]
        labels       = [b["obj_id"]       for b in batch]  # ground truth object id

        labels_tensor= torch.tensor(labels, dtype=torch.long, device=device) # [B]

        logits= self.forward(disease_texts, images) # => [B, num_objects]
        loss = self.loss_fn(logits, labels_tensor)

        # update accuracy
        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, labels_tensor)

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(batch))
        return loss

    def on_train_epoch_end(self):
        # log train_acc
        acc = self.train_acc.compute()
        self.log("train_acc", acc, prog_bar=True)
        self.train_acc.reset()

    # --- validation_step
    def validation_step(self, batch, batch_idx):
        device = self.device
        disease_texts= [b["disease_text"] for b in batch]
        images       = [b["image"]        for b in batch]
        labels       = [b["obj_id"]       for b in batch]

        labels_tensor= torch.tensor(labels, dtype=torch.long, device=device)
        logits= self.forward(disease_texts, images)
        loss= self.loss_fn(logits, labels_tensor)

        preds= logits.argmax(dim=1)
        self.val_acc.update(preds, labels_tensor)

        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch))

        # 可以在这里做log 3条到csv, 同样逻辑...
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        acc= self.val_acc.compute()
        self.log("val_acc", acc, prog_bar=True)
        self.val_acc.reset()

    # --- test_step
    def test_step(self, batch, batch_idx):
        device = self.device
        disease_texts= [b["disease_text"] for b in batch]
        images       = [b["image"]        for b in batch]
        labels       = [b["obj_id"]       for b in batch]

        labels_tensor= torch.tensor(labels, dtype=torch.long, device=device)
        logits= self.forward(disease_texts, images)
        loss= self.loss_fn(logits, labels_tensor)

        self.log("test_loss", loss, batch_size=len(batch))
        return {"test_loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def raw_collate_fn(batch):
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default=CSV_FILE)
    parser.add_argument("--image_dir", type=str, default=IMAGE_ROOT_DIR)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="fish_xlm_clip_vitmae_cbam_multiclass")

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--freeze_text", action="store_true", default=FREEZE_TEXT)
    parser.add_argument("--freeze_image", action="store_true", default=FREEZE_IMAGE)
    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = str(uuid.uuid4())[:8]
    run_dir = f"runs/run_{args.run_id}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("==== Configuration ====")
    print(f"CSV file:   {args.csv_file}")
    print(f"Image Dir:  {args.image_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"LR:         {args.lr}")
    print(f"Freeze Text:{args.freeze_text}")
    print(f"Freeze Img: {args.freeze_image}")
    print("=======================")

    # 1) 读取数据
    recs = load_triplets(args.csv_file)
    # 2) 收集所有object => 构建 object2id
    all_objects = list({r["object"] for r in recs})
    object2id   = {obj: i for i, obj in enumerate(all_objects)}
    num_objs    = len(object2id)
    print(f"[Info] total distinct objects: {num_objs}")

    random.shuffle(recs)
    split_idx = int(0.8 * len(recs))
    train_recs= recs[:split_idx]
    val_recs  = recs[split_idx:]
    test_recs = val_recs

    # 构建Dataset => MultiModalDataset
    train_ds = MultiModalDataset(train_recs, object2id=object2id)
    val_ds   = MultiModalDataset(val_recs,   object2id=object2id)
    test_ds  = MultiModalDataset(test_recs,  object2id=object2id)

    # 构建模型
    model = XLMClipViTMAECBAMModel(
        xlm_path=XLMR_MODEL_PATH,
        vitmae_path=VITMAE_MODEL_PATH,
        clip_path=CLIP_MODEL_PATH,
        lr=args.lr,
        freeze_text=args.freeze_text,
        freeze_image=args.freeze_image,
        num_objects=num_objs
    )

    # Dataloader
    train_loader= DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=raw_collate_fn
    )
    val_loader= DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=raw_collate_fn
    )
    test_loader= DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=raw_collate_fn
    )

    # Logger
    wandb_logger= WandbLogger(
        project=args.project_name,
        name=f"run_{args.run_id}",
        save_dir=run_dir
    )
    ckpt_callback= ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best",
        save_last=True
    )
    lr_monitor= LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer= pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=run_dir,
        logger=wandb_logger,
        callbacks=[ckpt_callback, lr_monitor],
        gradient_clip_val=1.0,
        precision=16 if torch.cuda.is_available() else 32,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10
    )

    print(f"Trainer device: {trainer.accelerator}, #devices={trainer.num_devices}")

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    print("Done. Best ckpt in:", ckpt_dir)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
