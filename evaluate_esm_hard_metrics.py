import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, roc_auc_score
import numpy as np

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
    def forward(self, x): return self.net(x)

# 加載數據與驗證集索引（與訓練一致）
data = torch.load("data/processed/esm_embeddings.pt")
X = data['embeddings'].numpy()
y = data['labels'].numpy()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, val_idx = next(sss.split(X, y))
X_val, y_true = X[val_idx], y[val_idx]

# 推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESM_MLP().to(device)
model.load_state_dict(torch.load("esm_mlp_best.pt", map_location=device))
model.eval()

with torch.no_grad():
    probs = model(torch.from_numpy(X_val).float().to(device)).cpu().numpy().ravel()

y_pred = (probs >= 0.5).astype(int)

# 指標
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
sens = recall_score(y_true, y_pred, zero_division=0)
spec = recall_score(1-y_true, 1-y_pred, zero_division=0)
mcc  = matthews_corrcoef(y_true, y_pred)
auc  = roc_auc_score(y_true, probs)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("驗證集硬指標（閾值0.5）：")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Sensitivity  : {sens:.4f}")
print(f"Specificity  : {spec:.4f}")
print(f"MCC          : {mcc:.4f}")
print(f"AUC          : {auc:.4f}")
print(f"混淆矩陣: TN={tn} FP={fp} FN={fn} TP={tp}")
