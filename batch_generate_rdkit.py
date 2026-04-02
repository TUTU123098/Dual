# batch_generate_rdkit.py
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

DATASETS = ["AOPP", "AnOxPP", "AnOxPePred"]
FOLDS = range(5)
BASE_CSV_DIR = "data"
BASE_RDKIT_DIR = "data/rdkit"

def generate_for_one_split(csv_path, out_dir):
    """为单个 train 或 test split 生成 compounds.pkl / edges.pkl / labels.npy"""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    compounds = []      # list of list or array, main.py 里转成 torch.tensor
    edges_list = []     # list of dict
    labels = []
    
    invalid = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {os.path.basename(csv_path)}"):
        seq = str(row['SEQUENCE']).strip().upper()
        label = float(row['label'])
        
        mol = Chem.MolFromSequence(seq)
        if mol is None or mol.GetNumAtoms() == 0:
            invalid += 1
            continue
            
        # 尽量贴近 main.py 期望的格式（参考你之前 grep 出的结构）
        # compounds: 这里简单用 [0]*n_atoms 占位（实际可换成原子类型 one-hot 等）
        n_atoms = mol.GetNumAtoms()
        compounds.append([0] * n_atoms)   # ← 如果 main.py 里 nodes_fp 只是占位，这就够
        
        # 边信息（最简单实现：化学键作为无向边）
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()  # 1.0 单键, 1.5 芳香, 2.0 双键, 3.0 三键
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([bond_type, bond_type])
        
        edge_index = np.array(edge_index).T if edge_index else np.empty((2, 0), dtype=int)
        edge_attr = np.array(edge_attr, dtype=float) if edge_attr else np.empty(0, dtype=float)
        
        edges_list.append({
            'edge_index_bond': edge_index.tolist(),
            'edge_attr_bond': edge_attr.tolist()
            # 如果 main.py 里用了 RBF，这里可以后续加 'edge_attr_rbf'
        })
        
        labels.append([label])
    
    print(f"无效序列数: {invalid} / {len(df)}")
    
    with open(os.path.join(out_dir, 'compounds.pkl'), 'wb') as f:
        pickle.dump(compounds, f)
    with open(os.path.join(out_dir, 'edges.pkl'), 'wb') as f:
        pickle.dump(edges_list, f)
    np.save(os.path.join(out_dir, 'labels.npy'), np.array(labels))
    
    print(f"完成保存 → {out_dir}")

# 主循环：生成所有 15 个 split
for ds in DATASETS:
    for fold in FOLDS:
        for split in ["train", "test"]:
            csv_path = os.path.join(BASE_CSV_DIR, ds, str(fold), f"{split}.csv")
            out_dir  = os.path.join(BASE_RDKIT_DIR, ds, str(fold), split)
            
            if not os.path.exists(csv_path):
                print(f"跳过不存在的文件: {csv_path}")
                continue
                
            print(f"\n正在处理: {ds} fold {fold} {split}")
            generate_for_one_split(csv_path, out_dir)

print("\n全部生成任务结束。请检查 data/rdkit 下是否有 15×3 = 45 个文件。")
