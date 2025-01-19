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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 使用 torchmetrics 进行多分类评估
from torchmetrics.classification import Accuracy
from transformers import PreTrainedTokenizer

from difflib import SequenceMatcher

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
#XLMR_MODEL_PATH     = "/home/shengguang/.cache/huggingface/hub/models--FacebookAI--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
#VITMAE_MODEL_PATH   = "/home/shengguang/.cache/huggingface/hub/models--facebook--vit-mae-base/snapshots/25b184bea5538bf5c4c852c79d221195fdd2778d"
#CLIP_MODEL_PATH     = "/home/shengguang/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

XLMR_MODEL_PATH     = "xlm-roberta-base"  # Hugging Face 模型名
VITMAE_MODEL_PATH   = "facebook/vit-mae-base"  # Hugging Face 模型名
CLIP_MODEL_PATH     = "openai/clip-vit-base-patch32"  # Hugging Face 模型名



####################################################
# Data Paths
####################################################
CSV_FILE       = r"C:\Users\yuyue\OneDrive\Documents\WeChat Files\wxid_0xz9g64tsb3b22\FileStorage\File\2025-01\FishGraph(1)\FishGraph\FishGraph\annotation.csv"
IMAGE_ROOT_DIR = r"C:\Users\yuyue\OneDrive\Documents\WeChat Files\wxid_0xz9g64tsb3b22\FileStorage\File\2025-01\FishGraph(1)\FishGraph\FishGraph\标注后的图像数据改名后的"

####################################################
# Step 1: read CSV & parse partial triplets
####################################################
def load_triplets(csv_file: str, image_root_dir: str):
    """
    读取annotation.csv，返回list[{...}]结构:
    {
        "image_path": ...,
        "text": ...,
        "disease": ...,
        "relation": ...,
        "object": ...
    }
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    # 加载 CSV 文件
    df = pd.read_csv(csv_file, encoding="utf-8")

    # 检查必需列是否存在
    required_cols = ["id", "image_path", "text", "病名", "关系", "感染对象"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"[Error] Missing column {c} in CSV!")

    # 转换为 triplets 格式
    triplets = []
    for _, row in df.iterrows():
        disease_name = str(row.get("病名", ""))
        relation = str(row.get("关系", ""))
        obj = row.get("感染对象", None)

        # 跳过没有 "感染对象" 的行
        if pd.isna(obj) or str(obj).strip() == "":
            continue

        rec = {
            "image_path": os.path.join(image_root_dir, str(row["image_path"]).replace("\\", "/")),
            "text": str(row["text"]),
            "disease": disease_name,
            "relation": relation,
            "object": str(obj).strip()
        }
        triplets.append(rec)

    return triplets


def create_prompt_examples():
    """
    生成插桩式 Prompt 数据，包括正例和负例。
    """
    prompt_examples = [
        {"text": "[LABEL]感染 [EXPLANATION]它可能是由病原体传播", "label": "感染"},
        {"text": "[LABEL]症状 [EXPLANATION]常见症状包括发烧和咳嗽", "label": "症状"},
        {"text": "[LABEL]流行季节 [EXPLANATION]这种疾病通常发生在夏季", "label": "流行季节"},
        {"text": "[LABEL]感染病 [EXPLANATION]这是不准确的写法", "label": None},
    ]
    return prompt_examples

def restrict_tokenizer_to_labels(tokenizer: PreTrainedTokenizer, label_words: list[str]):
    """
    限制 Tokenizer 的词表，仅包含指定的标签词。
    Args:
        tokenizer (PreTrainedTokenizer): 原始 Tokenizer。
        label_words (List[str]): 允许解码的标签词列表。
    Returns:
        PreTrainedTokenizer: 限制后的 Tokenizer。
    """
    vocab = tokenizer.get_vocab()
    allowed_tokens = {word: idx for word, idx in vocab.items() if word in label_words}

    # 保留指定的标签词，其他词用特殊标记替换（如 <unk>）
    tokenizer.vocab = allowed_tokens
    tokenizer.add_special_tokens({"additional_special_tokens": ["<unk>"]})
    return tokenizer

def augment_prompt_with_rare_classes(records, rare_classes=None):
    rare_classes = rare_classes or []
    augmented_records = []

    for record in records:
        augmented_records.append(record)
        # 针对稀有类增加示例
        if record["object"] in rare_classes:
            extra_prompt = {
                "disease_text": f"[LABEL]{record['relation']} [EXPLANATION]这是稀有类'{record['object']}'的示例。",
                "image_path": record["image_path"],
                "obj_id": record["obj_id"]
            }
            augmented_records.append(extra_prompt)
    return augmented_records

def label_discriminator(output_label, valid_labels, max_length=10):
    """
    判别器逻辑：
    1. 检查输出标签是否超出最大长度。
    2. 检查标签是否在合法词表中。
    3. 如果非法，基于相似度匹配最近的合法标签。
    """
    if len(output_label) > max_length:
        return None  # 非法输出

    if output_label not in valid_labels:
        # 相似度匹配纠正
        corrected_label = max(valid_labels, key=lambda label: cosine_similarity(output_label, label))
        return corrected_label

    return output_label

def truncate_output(generated_text, label_token="[LABEL]", max_tokens=3):
    """
    对生成的文本进行截断，限制 [LABEL] 后的 Token 数量。
    Args:
        generated_text (str): 模型生成的文本。
        label_token (str): 特殊标记，用于指示截断位置。
        max_tokens (int): [LABEL] 后允许的最大 Token 数。
    Returns:
        str: 截断后的文本。
    """
    if label_token not in generated_text:
        return generated_text  # 如果没有 [LABEL]，直接返回

    # 找到 [LABEL] 的位置并截取之后的文本
    label_index = generated_text.index(label_token)
    after_label = generated_text[label_index + len(label_token):].strip()

    # 按空格分割 Token
    tokens = after_label.split()
    if len(tokens) > max_tokens:
        truncated = " ".join(tokens[:max_tokens])
        return f"{generated_text[:label_index + len(label_token)]} {truncated}"
    
    return generated_text


def constrained_decoding_with_truncation(tokenizer, input_ids, model, label_words, max_tokens=3):
    """
    带约束的自适应截断解码，确保 [LABEL] 后只生成指定标签。
    
    Args:
        tokenizer: 用于分词和解码的 tokenizer 对象。
        input_ids: 输入的 token ids（PyTorch 张量）。
        model: Transformer 模型。
        label_words: 合法标签列表，例如 ['感染', '症状', '流行季节', ...]。
        max_tokens: [LABEL] 后允许生成的最大 token 数。
    
    Returns:
        Decoded string: 最终解码后的字符串。
    """
    device = input_ids.device
    valid_token_ids = tokenizer.convert_tokens_to_ids(label_words)  # 合法标签转为 token ids
    output_ids = input_ids.clone()  # 初始化解码输出，复制 input_ids

    # 初始模型状态
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            # 模型前向传播，获取 logits
            outputs = model(input_ids=output_ids)
            logits = outputs.logits[:, -1, :]  # 获取最后一个时间步的 logits

            # 设置非法 token 的概率为 -inf，确保模型只生成合法标签
            mask = torch.full_like(logits, fill_value=-float('inf'), device=device)
            mask[:, valid_token_ids] = logits[:, valid_token_ids]
            filtered_logits = mask

            # 选择概率最高的 token 作为下一个生成的 token
            next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)

            # 如果生成了 <eos> 或无效 token，则停止解码
            if next_token.item() == tokenizer.eos_token_id or next_token.item() not in valid_token_ids:
                break

            # 将生成的 token 拼接到输出序列中
            output_ids = torch.cat([output_ids, next_token], dim=-1)

    # 解码最终生成的序列
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded_output



def compute_loss_with_label_weighting(logits, labels, mask, weight_factor=2.0):
    """
    计算加权交叉熵损失，对 [LABEL] 部分提升权重。
    
    Args:
        logits: 模型输出的 logits，形状为 [B, num_classes]
        labels: 标签张量，形状为 [B]
        mask: 权重掩码，形状为 [B]
        weight_factor: 权重因子，用于控制 [LABEL] 的权重（默认 2.0）
    
    Returns:
        加权后的平均损失值
    """
    ce_loss = nn.CrossEntropyLoss(reduction='none')  # 不做平均，逐样本计算
    losses = ce_loss(logits, labels)  # [B]
    
    # 对 mask 中为 1 的样本提升权重
    weighted_loss = losses * (mask * (weight_factor - 1) + 1)
    
    # 返回加权后的平均损失
    return weighted_loss.mean()

def decode_label(self, label_idx):
    """
    将预测的索引转换为标签。
    """
    if hasattr(self, "id_to_label"):
        return self.id_to_label.get(label_idx, "UNKNOWN")
    else:
        raise ValueError("id_to_label mapping is missing.")

def oversample_data(records, target_column="object", rare_classes=None):
    """
    对少量类样本进行过采样。
    """
    rare_classes = rare_classes or []
    oversampled_records = []

    for record in records:
        if record[target_column] in rare_classes:
            # 根据稀有类的比例复制样本
            oversampled_records.extend([record] * 5)  # 5 是过采样倍数
        else:
            oversampled_records.append(record)

    random.shuffle(oversampled_records)
    return oversampled_records

def get_most_common_label(train_recs, object2id):
    label_counts = {k: 0 for k in object2id.keys()}
    for rec in train_recs:
        label_counts[rec["object"]] += 1
    most_common_label = max(label_counts, key=label_counts.get)
    return object2id[most_common_label]



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
    增加双通道⽂本⽣成函数
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
        label_text, explanation_text = self._process_generated_output(rec['text'])
         # 合并 disease 和 relation 与 label_text 作为输入
        if label_text:
            disease_text = f"{rec['disease']} {rec['relation']} {label_text}"
        else:
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
            "explanation_text": explanation_text,
            "relation": rec["relation"],
            "text": rec['text'],
            "image": image,
            "obj_id": obj_id
        }

    def _process_generated_output(self, text):
        """
        内部方法：解析 [LABEL] 和 [EXPLANATION]
        """
        if "[LABEL]" in text:
            parts = text.split("[EXPLANATION]")
            label_part = parts[0].replace("[LABEL]", "").strip()
            explanation_part = parts[1].strip() if len(parts) > 1 else ""
            return label_part, explanation_part
        return None, text  # 返回默认值

    def augment_with_prompts(data, prompt_examples):
        """
        将插桩式 Prompt 数据整合到训练数据中。
        """
        return data + prompt_examples


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
        num_objects=100,  # 所有 object 的总数
        valid_labels=["感染", "症状", "流行季节", "流行地区"],
        scheduler_type="cosine",  
        object2id=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.scheduler_type = scheduler_type
        self.lr = lr
        self.num_objects = num_objects
        self.valid_labels = valid_labels or []
        self.object2id = object2id  # 保存 object2id
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_path)

        # 1) XLM-R
      #  self.xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_path)
      #  self.xlm_model     = XLMRobertaModel.from_pretrained(xlm_path)
      #  self.xlm_hidden_size = self.xlm_model.config.hidden_size

        # 2) CLIP
      #  self.clip_model = CLIPModel.from_pretrained(clip_path)
      #  self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
      #  self.clip_proj_dim = self.clip_model.config.projection_dim

        # 3) ViTMAE
      #  self.vitmae_model = ViTMAEModel.from_pretrained(vitmae_path)
      #  self.vitmae_hidden_size = self.vitmae_model.config.hidden_size
      
        # 从 Hugging Face Hub 加载预训练模型
        self.xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_path)
        self.xlm_model     = XLMRobertaModel.from_pretrained(xlm_path)
        self.xlm_hidden_size = self.xlm_model.config.hidden_size


        self.clip_model     = CLIPModel.from_pretrained(clip_path)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
        self.clip_proj_dim = self.clip_model.config.projection_dim

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
    
    def edit_distance(self, a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    def map_to_valid_label(self, output_label, valid_labels):
        if output_label in valid_labels:
            return output_label
        
        # 使用编辑距离找到最接近的合法标签
        closest_label = min(valid_labels, key=lambda label: self.edit_distance(output_label, label))
        return closest_label
    
    
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

        # 1. 格式化输入 Prompt
        # 使用 [LABEL] 和 [EXPLANATION] 格式的 Prompt
        #print(f"Batch sample: {batch[0]}") 
        disease_texts = [
            f"[LABEL] {b['relation']} [EXPLANATION] {b['text']} "
            f"请确保 [LABEL] 的输出仅包含以下集合中的词：{', '.join(self.valid_labels)}"
        for b in batch
        ]
        images = [b["image"] for b in batch]
        labels = [b["obj_id"] for b in batch]  # ground truth object id

         # 2. 将标签转为 Tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)  # [B]

         # 3. 前向传播
        logits = self.forward(disease_texts, images)  # => [B, num_objects]

        # 4. 定义 mask，用于提高 [LABEL] 部分的权重
        label_mask = torch.tensor(
          [1 if "[LABEL]" in txt else 0 for txt in disease_texts], dtype=torch.float32, device=device
        )

         # 5. 自定义 Loss 函数，提升 [LABEL] 后的 Token 权重
        loss = compute_loss_with_label_weighting(logits, labels_tensor, label_mask)

        # 6. 更新准确率
        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, labels_tensor)

        # 7. 记录训练损失
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

        # 提取输入数据
        disease_texts = [b["disease_text"] for b in batch]
        images = [b["image"] for b in batch]
        labels = [b["obj_id"] for b in batch]

         # 转换标签为张量
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

        # 前向计算 logits
        logits = self.forward(disease_texts, images)
        loss = self.loss_fn(logits, labels_tensor)

        # 获取预测结果及其置信度
        probs = torch.softmax(logits, dim=1)  # 计算每个类别的概率
        preds = logits.argmax(dim=1)  # 获取预测标签索引
        confidences = probs.max(dim=1).values  # 获取每个预测的最高置信度

        # 更新验证准确率
        self.val_acc.update(preds, labels_tensor)

         # 记录验证损失
        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch))

        # 添加 [LABEL] 通道受限解码逻辑
        valid_labels = ["感染", "症状", "流行季节", "流行地区"]  # 定义合法标签集合
        
        object2id = {label: idx for idx, label in enumerate(valid_labels)}
        print(f"[Info] object2id mapping: {object2id}")
        decoded_labels = [
           valid_labels[pred] if pred < len(valid_labels) else "未知"
           for pred in preds.cpu().numpy()
        ]

        print(decoded_labels)

         # 结合置信度和标签
        results_with_confidence = [
            {"decoded_label": decoded_labels[i], "confidence": confidences[i].item()}
             for i in range(len(decoded_labels))
           ]
     
        #filtered_results = [
        #r for r in results_with_confidence if r["confidence"] >= 0.8
        #] 可以根据置信度阈值（例如 0.8）过滤低置信度标签

        # 输出示例结果
        if batch_idx == 0:  # 仅记录第一批次
          print(f"Sample decoded outputs with confidence: {results_with_confidence[:3]}")

        # 返回验证损失和预测结果
        return {
           "val_loss": loss,
           "decoded_labels": decoded_labels,
           "confidences": confidences.cpu().numpy(),
         }


    def on_validation_epoch_end(self):
        acc= self.val_acc.compute()
        self.log("val_acc", acc, prog_bar=True)
        self.val_acc.reset()

    # --- test_step
    def test_step(self, batch, batch_idx):
        device = self.device
        disease_texts = [b["disease_text"] for b in batch]
        images = [b["image"] for b in batch]
        labels = [b["obj_id"] for b in batch]

        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        logits = self.forward(disease_texts, images)

        label_words = ["感染", "症状", "流行季节", "流行地区", "温度"]

        # Generate predicted labels
        preds = logits.argmax(dim=1)

        max_tokens = 2

        # 转换为 input_ids
        input_ids = self.xlm_tokenizer(disease_texts, return_tensors="pt", padding=True).input_ids.to(device)
        # 执行约束解码
        result = constrained_decoding_with_truncation(
                   tokenizer=self.tokenizer,
                   input_ids=input_ids,
                   model=self.xlm_model,  
                   label_words=label_words,
                   max_tokens=max_tokens
                 )

        print("Constrained decoding result:", result)
        
        loss = self.loss_fn(logits, labels_tensor)
        self.log("test_loss", loss, batch_size=len(batch))

        # 记录批次信息
        self.log("batch_idx", batch_idx)

        
        # Return mapped labels and loss
        return {
            "test_loss": loss, 
            "batch_idx": batch_idx  # 添加到返回值中
        }

 


    def configure_optimizers(self):
        print("Scheduler type:", getattr(self, "scheduler_type", "Not found"))
        if not hasattr(self, 'scheduler_type'):
            raise ValueError("scheduler_type attribute is missing!")

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        if self.scheduler_type == "cosine":
           scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, eta_min=0)
        elif self.scheduler_type == "step":
           scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        else:
           raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        return [optimizer], [scheduler]

def raw_collate_fn(batch):
    return batch

def cosine_similarity(str1, str2):
    """简单的余弦相似度实现"""
    set1 = set(str1)
    set2 = set(str2)
    intersection = set1.intersection(set2)
    if not set1 or not set2:
        return 0.0
    return len(intersection) / (len(set1) * len(set2)) ** 0.5

class Trainer:
    def __init__(self, model, scheduler: str = 'cosine', *args, **kwargs):
        if scheduler == "cosine":
            scheduler_type = "cosine"
        elif scheduler == "step":
            scheduler_type = "step"
        else:
            scheduler_type = "unknown"
        
        # 传递给模型
        model.scheduler_type = scheduler_type
        self.model = model

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

    # 1) 读取原始数据
    recs = load_triplets(args.csv_file, args.image_dir)

    # 定义稀有类标签
    rare_classes = ["温度", "时间"]

    # 2) 添加插桩式 Prompt 数据
    prompt_examples = create_prompt_examples()  # 调用插桩函数
    recs = augment_prompt_with_rare_classes(recs, prompt_examples)  # 数据增强

    # 3) 收集所有object => 构建 object2id
    all_objects = list({r["object"] for r in recs})
    object2id = {obj: i for i, obj in enumerate(all_objects)}
    num_objs = len(object2id)
    print(f"[Info] total distinct objects: {num_objs}")

    # 4) 数据拆分
    random.shuffle(recs)

     # 划分测试集：先取出 10% 数据作为测试集
    test_ratio = 0.1
    test_size = int(test_ratio * len(recs))
    test_recs = recs[:test_size]
    remaining_recs = recs[test_size:]  
    val_ratio = 0.3  
    val_size = int(val_ratio * len(remaining_recs))
    val_recs = remaining_recs[:val_size]
    train_recs = remaining_recs[val_size:] 


    # 过采样少量类
    train_recs = oversample_data(train_recs, target_column="object", rare_classes=rare_classes)

    # 构建Dataset => MultiModalDataset
    train_ds = MultiModalDataset(train_recs, object2id=object2id)
    val_ds   = MultiModalDataset(val_recs,   object2id=object2id)
    test_ds  = MultiModalDataset(test_recs,  object2id=object2id)

    valid_labels = ["感染", "症状", "流行季节", "流行地区"]

    # 构建模型
    model = XLMClipViTMAECBAMModel(
        xlm_path=XLMR_MODEL_PATH,
        vitmae_path=VITMAE_MODEL_PATH,
        clip_path=CLIP_MODEL_PATH,
        lr=args.lr,
        freeze_text=args.freeze_text,
        freeze_image=args.freeze_image,
        num_objects=num_objs,
        valid_labels=valid_labels,  # 将标签传入模型
        object2id=object2id
    )

    # Dataloader
    train_loader= DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=raw_collate_fn
    )
    val_loader= DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=raw_collate_fn
    )
    test_loader= DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
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
