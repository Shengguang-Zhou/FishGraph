#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import argparse
from collections import Counter

################################################################################
# Configuration placeholders
################################################################################
DEFAULT_CSV = "/home/shengguang/PycharmProjects/movielens_recommendation/FishGraph/FishGraph/annotation.csv"   # or your actual CSV path
DEFAULT_IMAGE_ROOT = "/home/shengguang/PycharmProjects/movielens_recommendation/FishGraph/FishGraph/标注后的图像数据改名后的"

################################################################################
# parse_args
################################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default=DEFAULT_CSV,
                        help="Path to the CSV annotation file.")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_ROOT,
                        help="Root directory of images.")
    return parser.parse_args()

################################################################################
# load_csv
################################################################################
def load_csv(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file, encoding="utf-8")
    return df

################################################################################
# Basic checks
################################################################################
def basic_dataset_checks(df):
    """
    Example checks:
     1) required columns
     2) total rows
     3) distinct objects
     4) distribution
     etc.
    """
    required_cols = ["id", "image_path", "text", "病名", "关系", "感染对象"]
    for c in required_cols:
        if c not in df.columns:
            print(f"[WARNING] Missing required column: {c}")
    print(f"[INFO] Total rows in CSV: {len(df)}")

    # gather objects
    if "感染对象" not in df.columns:
        print("[ERROR] '感染对象' column not present.")
        return

    # Filter out rows missing the object
    df = df.dropna(subset=["感染对象"])
    df["感染对象"] = df["感染对象"].astype(str).str.strip()

    # Count distinct objects
    all_objs = df["感染对象"].unique().tolist()
    print(f"[INFO] Distinct objects count: {len(all_objs)}")

    # Frequency distribution
    obj_freq = Counter(df["感染对象"].tolist())
    print("[INFO] Object frequency distribution (top 20):")
    for obj, freq in obj_freq.most_common(20):
        print(f"   {obj} => {freq}")

    # Possibly check if some objects have very few occurrences
    small_objs = [o for o,f in obj_freq.items() if f < 3]  # threshold=3
    if small_objs:
        print(f"[INFO] Found {len(small_objs)} objects with fewer than 3 samples. "
               "Consider merging or removing them if performance is too poor.")

    # Could also show relation distribution
    if "关系" in df.columns:
        rel_freq = Counter(df["关系"].tolist())
        print("[INFO] Relation frequency distribution:")
        for rel, freq in rel_freq.most_common():
            print(f"   {rel} => {freq}")

    return df

################################################################################
# train/val/test splits
################################################################################
def train_val_test_split(df, split_ratio=0.8):
    """
    A naive example of random splitting:
      80% train, 10% val, 10% test
    or whatever ratio you want.
    If you already have a column for splitting, adapt accordingly.
    """
    # shuffle
    df_shuf = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df_shuf)
    train_n = int(split_ratio * n)
    # If you prefer 80/10/10, do something else. For demonstration:
    #   train => first 80%
    #   val => next 10%
    #   test => last 10%
    train_df = df_shuf.iloc[:train_n]
    remain_df = df_shuf.iloc[train_n:]
    val_n = len(remain_df)//2
    val_df  = remain_df.iloc[:val_n]
    test_df = remain_df.iloc[val_n:]
    return train_df, val_df, test_df

def analyze_splits(train_df, val_df, test_df):
    # Distinct objects in each
    train_objs = set(train_df["感染对象"].unique().tolist())
    val_objs   = set(val_df["感染对象"].unique().tolist())
    test_objs  = set(test_df["感染对象"].unique().tolist())

    print(f"\n[Splits Analysis]")
    print(f"Train: {len(train_df)} samples, {len(train_objs)} distinct objs")
    print(f"Val:   {len(val_df)} samples,   {len(val_objs)} distinct objs")
    print(f"Test:  {len(test_df)} samples,  {len(test_objs)} distinct objs")

    # Check if val or test have objects never seen in train
    val_unseen  = val_objs - train_objs
    test_unseen = test_objs - train_objs
    if val_unseen:
        print(f"[WARNING] {len(val_unseen)} objects found in val that never appear in train!")
        print("   ", val_unseen)
    if test_unseen:
        print(f"[WARNING] {len(test_unseen)} objects found in test that never appear in train!")
        print("   ", test_unseen)

################################################################################
# main
################################################################################
def main():
    args = parse_args()

    df = load_csv(args.csv_file)
    df_checked = basic_dataset_checks(df)

    # do naive split
    train_df, val_df, test_df = train_val_test_split(df_checked, split_ratio=0.8)
    analyze_splits(train_df, val_df, test_df)

    # Additional checks:
    # e.g. how many images are missing?
    missing_count = 0
    for _, row in df_checked.iterrows():
        imgp = os.path.join(args.image_dir, str(row["image_path"]).replace("\\","/"))
        if not os.path.exists(imgp):
            missing_count += 1
    if missing_count>0:
        print(f"[WARNING] Found {missing_count} image paths that don't exist in image_dir: {args.image_dir}")
    else:
        print("[INFO] All images exist in the specified directory (at least by path).")

    print("\n[DONE] Dataset checks completed.\n")

if __name__ == "__main__":
    main()
