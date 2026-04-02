# make_folds.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import json

DATA_PATH = "data/combined/combined_data.csv"
OUTPUT_DIR = "data/folds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"总样本数：{len(df)}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_summary = []

for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['label'])):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)
    
    fold_dict = {
        "fold": fold,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_pos": int(train_df['label'].sum()),
        "test_pos": int(test_df['label'].sum()),
    }
    fold_summary.append(fold_dict)
    
    # 保存切分文件（方便后续直接用路径加载）
    train_path = f"{OUTPUT_DIR}/fold_{fold}_train.csv"
    test_path  = f"{OUTPUT_DIR}/fold_{fold}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Fold {fold+1:2d} | train: {len(train_df):4d}  test: {len(test_df):4d}")

# 保存总结
with open(f"{OUTPUT_DIR}/fold_summary.json", "w") as f:
    json.dump(fold_summary, f, indent=2)

print(f"\n5折文件已生成在：{OUTPUT_DIR}")
print("每个 fold 的 test 集大小应接近 1047 条")
