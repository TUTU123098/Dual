import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             matthews_corrcoef, roc_auc_score, roc_curve)
import esm

# ── 配置 ─────────────────────────────────────────────
DATASETS   = ["AOPP", "AnOxPP", "AnOxPePred"]
BASE_DIR   = "data"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 50
PATIENCE   = 12
LR         = 1e-4
WEIGHT_DECAY = 1e-2
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 加载 ESM ─────────────────────────────────────────
print("加载 ESM 模型...")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.to(DEVICE).eval()
batch_converter = alphabet.get_batch_converter()
print("ESM 加载完成")

# ── 提取 embedding ────────────────────────────────────
@torch.inference_mode()
def extract_embeddings(sequences, batch_size=8):
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEVICE)
        results = esm_model(tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33]
        for j, seq in enumerate(batch_seqs):
            emb = token_reps[j, 1:len(seq)+1].mean(0)
            all_embs.append(emb.cpu())
    return torch.stack(all_embs)

# ── MLP 分类头 ─────────────────────────────────────────
class ESM_MLP(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),   nn.ReLU(), nn.Dropout(0.1),
            nn.LayerNorm(512),
            nn.Linear(512, 128),    nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(128, 1),      nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ── 单折训练 ───────────────────────────────────────────
def train_one_fold(dataset_name, fold):
    print(f"\n{'='*50}")
    print(f"{dataset_name}  Fold {fold}")
    print(f"{'='*50}")

    train_path = os.path.join(BASE_DIR, dataset_name, str(fold), "train.csv")
    test_path  = os.path.join(BASE_DIR, dataset_name, str(fold), "test.csv")

    if not os.path.exists(train_path):
        print(f"跳过：{train_path} 不存在")
        return None

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"提取训练集 embedding ({len(train_df)} 条)...")
    X_train = extract_embeddings(train_df["SEQUENCE"].tolist())
    y_train = torch.tensor(train_df["label"].values, dtype=torch.float32).unsqueeze(1)

    print(f"提取测试集 embedding ({len(test_df)} 条)...")
    X_test  = extract_embeddings(test_df["SEQUENCE"].tolist())
    y_test  = torch.tensor(test_df["label"].values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=512)

    model     = ESM_MLP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()

    best_auc   = -1
    best_preds = None
    best_trues = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds.append(model(xb.to(DEVICE)).cpu())
                trues.append(yb)

        preds = torch.cat(preds).numpy().ravel()
        trues = torch.cat(trues).numpy().ravel()
        auc   = roc_auc_score(trues, preds)

        print(f"Epoch {epoch+1:3d}  loss={total_loss/len(train_loader):.4f}  AUC={auc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_preds = preds
            best_trues = trues
            no_improve = 0
            torch.save(model.state_dict(), f"results/best_{dataset_name}_fold{fold}.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping")
                break

    fpr, tpr, th = roc_curve(best_trues, best_preds)
    best_thresh   = th[np.argmax(tpr - fpr)]
    y_pred        = (best_preds >= best_thresh).astype(int)

    metrics = {
        "Dataset": dataset_name,
        "Fold":    fold,
        "AUC":     best_auc,
        "Acc":     accuracy_score(best_trues, y_pred),
        "Prec":    precision_score(best_trues, y_pred, zero_division=0),
        "Sens":    recall_score(best_trues, y_pred, zero_division=0),
        "Spec":    recall_score(1-best_trues, 1-y_pred, zero_division=0),
        "MCC":     matthews_corrcoef(best_trues, y_pred)
    }
    print(f"Fold {fold} 结果: {metrics}")
    return metrics

# ── 主循环 ─────────────────────────────────────────────
all_fold_metrics = []

for ds in DATASETS:
    ds_results = []
    for fold in range(5):
        res = train_one_fold(ds, fold)
        if res:
            ds_results.append(res)
            all_fold_metrics.append(res)

    if ds_results:
        df_res = pd.DataFrame(ds_results)
        print(f"\n{'='*50}")
        print(f"{ds}  5折汇总")
        print(f"{'='*50}")
        for col in ["AUC","Acc","Prec","Sens","Spec","MCC"]:
            print(f"{col:6}: {df_res[col].mean():.4f} ± {df_res[col].std():.4f}")

# ── 保存所有结果 ───────────────────────────────────────
detail_df = pd.DataFrame(all_fold_metrics)
detail_df.to_csv("results/all_fold_results.csv", index=False)

# 汇总均值±标准差
summary_rows = []
for ds in DATASETS:
    sub = detail_df[detail_df["Dataset"] == ds]
    row = {"Dataset": ds}
    for col in ["AUC","Acc","Prec","Sens","Spec","MCC"]:
        row[f"{col}_mean"] = sub[col].mean()
        row[f"{col}_std"]  = sub[col].std()
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results/summary.csv", index=False)

print("\n✅ 结果已保存：")
print("   results/all_fold_results.csv  （每折详细指标）")
print("   results/summary.csv           （每个数据集均值±标准差）")
print("\n全部完成！")
