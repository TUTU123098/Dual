import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np

# 加载预计算的 ESM embedding
data = torch.load("data/processed/esm_embeddings.pt")
X = data['embeddings']          # (5235, 1280)
y = data['labels'].unsqueeze(1) # (5235, 1)

# 简单 80/20 划分（先不做5折，快速看效果）
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

# 极简三层 MLP
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = SimpleMLP()
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"训练开始，使用 {device}")

best_auc = 0
for epoch in range(30):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 验证
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            preds.append(out.cpu())
            trues.append(yb.cpu())
    
    preds = torch.cat(preds).numpy().ravel()
    trues = torch.cat(trues).numpy().ravel()
    auc = roc_auc_score(trues, preds)
    
    print(f"Epoch {epoch+1:2d} | train_loss: {train_loss/len(train_loader):.4f} | val_auc: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "esm_mlp_best.pt")

print(f"最高 val AUC: {best_auc:.4f}")
