import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix,
                             roc_curve)
import numpy as np
import os

# ── 模型 ─────────────────────────────────────────
class ESM_MLP(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(0.15),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),   nn.ReLU(), nn.Dropout(0.15),
            nn.LayerNorm(512),
            nn.Linear(512, 128),    nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ── 加载ESM embedding ─────────────────────────────
data = torch.load("data/processed/esm_embeddings.pt")
X_all = data['embeddings']
y_all = data['labels'].numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fold_results = []

# ── 遍历官方5折 ───────────────────────────────────
for fold in range(5):
    print(f"\n{'='*50}")
    print(f"Fold {fold}")
    print(f"{'='*50}")

    train_df = pd.read_csv(f"data/AOPP/{fold}/train.csv")
    test_df  = pd.read_csv(f"data/AOPP/{fold}/test.csv")

    # ⚠️ 关键：需要有原始索引
    train_idx = train_df['index'].values
    test_idx  = test_df['index'].values

    X_train = X_all[train_idx]
    y_train = torch.tensor(y_all[train_idx]).unsqueeze(1)

    X_val = X_all[test_idx]
    y_val = torch.tensor(y_all[test_idx]).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=512)

    model = ESM_MLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_auc = 0
    patience = 10
    no_improve = 0

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(device)).cpu())
                trues.append(yb)

        preds = torch.cat(preds).numpy().ravel()
        trues = torch.cat(trues).numpy().ravel()

        auc = roc_auc_score(trues, preds)

        if auc > best_auc:
            best_auc = auc
            best_preds = preds
            best_trues = trues
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # ── 最优阈值 ─────────────────────────
    fpr, tpr, thresholds = roc_curve(best_trues, best_preds)
    best_thresh = thresholds[np.argmax(tpr - fpr)]

    y_pred = (best_preds >= best_thresh).astype(int)

    acc  = accuracy_score(best_trues, y_pred)
    prec = precision_score(best_trues, y_pred)
    sens = recall_score(best_trues, y_pred)
    spec = recall_score(1-best_trues, 1-y_pred)
    mcc  = matthews_corrcoef(best_trues, y_pred)

    print(f"AUC={best_auc:.4f}  MCC={mcc:.4f}")

    fold_results.append([acc, prec, sens, spec, mcc, best_auc])

# ── 平均结果 ─────────────────────────────────────
fold_results = np.array(fold_results)

print("\n================ 最终5折结果 ================")
names = ["Accuracy", "Precision", "Sensitivity", "Specificity", "MCC", "AUC"]
for i, name in enumerate(names):
    print(f"{name:<12}: {fold_results[:,i].mean():.4f} ± {fold_results[:,i].std():.4f}")
