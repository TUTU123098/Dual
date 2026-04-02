# prepare_rdkit_graph.py
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import numpy as np

CSV_PATH    = "data/combined/combined_data.csv"
OUTPUT_DIR  = "data/processed/rdkit_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("读取数据...")
df = pd.read_csv(CSV_PATH)
print(f"总序列数：{len(df)}")

invalid_count = 0
node_counts   = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成 RDKit 分子图"):
    seq = str(row['SEQUENCE']).strip().upper()
    label = row['label']

    mol = Chem.MolFromSequence(seq)
    if mol is None or mol.GetNumAtoms() == 0:
        print(f"  跳过无效序列 (idx={idx}): {seq}")
        invalid_count += 1
        continue

    # 计算 2D 坐标（短肽足够）
    try:
        AllChem.Compute2DCoords(mol)
    except Exception as e:
        print(f"  坐标生成失败 (idx={idx}): {seq} → {e}")
        continue

    # 节点特征（9 维，常见且足够）
    atom_feats = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),                    # 0
            atom.GetTotalDegree(),                  # 1
            atom.GetFormalCharge(),                 # 2
            atom.GetTotalNumHs(),                   # 3
            int(atom.GetIsAromatic()),              # 4
            int(atom.IsInRing()),                   # 5
            atom.GetMass() / 100.0,                 # 6 归一化
            atom.GetExplicitValence(),              # 7
            int(atom.GetHybridization() or 0),      # 8
        ]
        atom_feats.append(feat)

    x = torch.tensor(atom_feats, dtype=torch.float)  # (N, 9)

    # 边索引与边特征（3 维）
    edge_index = []
    edge_attr  = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])

        btype   = bond.GetBondTypeAsDouble()
        conj    = int(bond.GetIsConjugated())
        in_ring = int(bond.IsInRing())

        attr = [btype, conj, in_ring]
        edge_attr.extend([attr, attr])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(
        x           = x,
        edge_index  = edge_index,
        edge_attr   = edge_attr,
        y           = torch.tensor([label], dtype=torch.float),
        seq         = seq,
        num_nodes   = x.size(0),
        original_idx = idx,          # 用于后续匹配 ESM embedding
    )

    torch.save(data, f"{OUTPUT_DIR}/graph_{idx:05d}.pt")
    node_counts.append(x.size(0))

# ── 总结 ─────────────────────────────────────────────────────────────
print(f"\n生成完成")
print(f"有效分子图：{len(node_counts)} / {len(df)}")
print(f"无效/跳过数量：{invalid_count}")

if node_counts:
    print("\n节点数（原子数）统计：")
    print(f"  最小：{min(node_counts)}")
    print(f"  最大：{max(node_counts)}")
    print(f"  平均：{np.mean(node_counts):.2f}")
    print(f"  中位数：{np.median(node_counts):.0f}")

print(f"\n所有图文件保存至：{OUTPUT_DIR}")
print("下一步：修改 AOPDataset 同时加载 ESM + 图")