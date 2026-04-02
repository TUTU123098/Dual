# precompute_esm.py
# 对 5235 条序列批量提取 ESM-2 mean-pooled embedding，保存为 .pt 文件

import pandas as pd
import torch
from esm import pretrained
from tqdm import tqdm
import os

# ── 路径配置 ──────────────────────────────────────────────────
CSV_PATH   = "data/combined/combined_data.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_PT  = f"{OUTPUT_DIR}/esm_embeddings.pt"   # 最终保存路径
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 设备选择 ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# ── 加载 ESM-2 ────────────────────────────────────────────────
print("加载 ESM-2 模型...")
model, alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()
print("模型加载完成")

# ── 读取数据 ──────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"总序列数：{len(df)}")

# ── 批量提取（batch_size 可根据显存调整） ─────────────────────
BATCH_SIZE = 32   # CPU 上建议 16~32；GPU 可以更大

all_embeddings = []
all_labels     = []

# 把 df 按 BATCH_SIZE 分组处理
sequences = df['SEQUENCE'].tolist()
labels    = df['label'].tolist()

for batch_start in tqdm(range(0, len(sequences), BATCH_SIZE), desc="提取 ESM embedding"):
    batch_seqs   = sequences[batch_start : batch_start + BATCH_SIZE]
    batch_labels = labels[batch_start : batch_start + BATCH_SIZE]

    # fair-esm 需要 (name, sequence) 格式
    batch_input = [(str(i), seq.strip().upper()) for i, seq in enumerate(batch_seqs)]

    batch_labels_esm, batch_strs, batch_tokens = batch_converter(batch_input)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    # token_reps: (B, L+2, 1280)
    token_reps = results["representations"][33]   # (B, L_max+2, 1280)

    for i, seq in enumerate(batch_seqs):
        L = len(seq)
        # 去掉 <cls>(位置0) 和 <eos>(位置L+1)，取真实残基部分
        residue_emb = token_reps[i, 1 : L + 1]    # (L, 1280)
        mean_emb    = residue_emb.mean(dim=0).cpu() # (1280,)
        all_embeddings.append(mean_emb)

    all_labels.extend(batch_labels)

# ── 拼接并保存 ────────────────────────────────────────────────
all_embeddings_tensor = torch.stack(all_embeddings)   # (5235, 1280)
all_labels_tensor     = torch.tensor(all_labels, dtype=torch.float32)  # (5235,)

print(f"\nEmbedding 矩阵形状：{all_embeddings_tensor.shape}")
print(f"Label 向量形状：   {all_labels_tensor.shape}")
print(f"正样本数：{all_labels_tensor.sum().int().item()}")

torch.save({
    'embeddings': all_embeddings_tensor,   # (5235, 1280)
    'labels':     all_labels_tensor,       # (5235,)
    'sequences':  sequences,               # list[str]，保留原始序列方便 debug
}, OUTPUT_PT)

print(f"\n已保存至：{OUTPUT_PT}")
print("下一步：运行 python train_baseline.py 验证分类效果")
