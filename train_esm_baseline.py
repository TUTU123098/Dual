import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import numpy as np

# ── 加载预计算的 ESM embedding ────────────────────────────────────────
print("加载 ESM embedding...")
data = torch.load("data/processed/esm_embeddings.pt")
X = data['embeddings']                    # (5235, 1280)
y = data['labels'].unsqueeze(1)           # (5235, 1)

print(f"总样本: {len(X)} | 正样本比例: {y.mean().item():.4f}")

# ── 分层切分（保证 val 集正负均衡） ────────────────────────────────────
print("\n使用分层随机切分...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(X, y.squeeze().numpy()))

X_train = X[train_idx]
y_train = y[train_idx]
X_val   = X[val_idx]
y_val   = y[val_idx]

print(f"训练集: {len(y_train)} 条 | 正比例: {y_train.mean().item():.4f}")
print(f"验证集: {len(y_val)}   条 | 正比例: {y_val.mean().item():.4f}")

# 转成 Dataset + DataLoader
train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)

# ── 模型：加厚 + LayerNorm ──────────────────────────────────────────────
class ESM_MLP(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ── 训练设置 ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESM_MLP().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

print(f"\n开始训练，使用设备：{device}\n")

best_auc = 0.0
patience = 12
no_improve = 0

for epoch in range(60):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # 验证
    model.eval()
    preds, trues = [], []
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            val_loss += criterion(out, yb.to(device)).item() * xb.size(0)
            preds.append(out.cpu())
            trues.append(yb.cpu())

    val_loss /= len(val_loader.dataset)
    preds = torch.cat(preds).numpy().ravel()
    trues = torch.cat(trues).numpy().ravel()
    auc = roc_auc_score(trues, preds) if len(np.unique(trues)) > 1 else float('nan')

    print(f"Epoch {epoch+1:2d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_auc: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "esm_mlp_best.pt")
        no_improve = 0
        print(f"  ↑ 最佳 AUC 更新: {best_auc:.4f}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"早停触发 ( patience={patience} )")
            break

print(f"\n训练结束")
print(f"最高验证 AUC: {best_auc:.4f}")
print(f"最佳模型已保存至：esm_mlp_best.pt")
