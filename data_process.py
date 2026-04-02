# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import json
import pickle
import numpy as np
from rdkit import Chem
import pandas as pd

# =========================
# 配置
# =========================
RADIUS = 2
DIR_BASE = 'Data/rdkit/radius2'
DIR_TRAIN = os.path.join(DIR_BASE, 'train')
DIR_TEST  = os.path.join(DIR_BASE, 'test')
DIR_INDEP = os.path.join(DIR_BASE, 'independent_test')

os.makedirs(DIR_TRAIN, exist_ok=True)
os.makedirs(DIR_TEST,  exist_ok=True)
os.makedirs(DIR_INDEP, exist_ok=True)

# UNK 标记
ATOM_UNK = '__UNK_ATOM__'
BOND_UNK = '__UNK_BOND__'
EDGE_UNK = '__UNK_EDGE__'
FP_UNK   = '__UNK_FP__'

# =========================
# 词典与索引器
# =========================
def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)

def load_dictionary(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class GrowingIndexer:
    """训练阶段用: 未见过就分配新 id"""
    def __init__(self, backing_dict):
        self.d = backing_dict
    def id(self, key):
        if key not in self.d:
            self.d[key] = len(self.d)
        return self.d[key]

class FrozenIndexer:
    """测试阶段用: 未见过就映射到 UNK"""
    def __init__(self, backing_dict, unk_key):
        self.d = dict(backing_dict)
        if unk_key not in self.d:
            self.d[unk_key] = len(self.d)
        self.unk_id = self.d[unk_key]
    def id(self, key):
        return self.d.get(key, self.unk_id)

# =========================
# 图构建工具
# =========================
def create_atoms(mol, atom_indexer):
    """原子类别 含芳香性标记"""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAtoms():
        if a.GetIsAromatic():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_indexer.id(a) for a in atoms]
    return np.array(atoms, dtype=np.int64)

def create_ijbonddict(mol, bond_indexer):
    """每个节点的邻居 与‘键类型 id’"""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_indexer.id(str(b.GetBondType()))
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def wl_update_nodes_edges(atoms, i_jbond_dict, radius, fp_indexer, edge_indexer):
    """
    WL 半径 r 的迭代:
      - nodes: 节点标签迭代
      - i_jedge_dict: 基于“当前节点标签 + 键类型”得到的边标签迭代
    返回:
      nodes_fp: 最终节点指纹 id，shape [N]
      i_jedge_dict: dict[int -> list[(nbr, edge_fp_id)]]
    """
    if (len(atoms) == 1) or (radius == 0):
        nodes_fp = np.array([fp_indexer.id(a) for a in atoms], dtype=np.int64)
        i_jedge_dict = defaultdict(lambda: [])
        for i, j_edge in i_jbond_dict.items():
            for j, bond in j_edge:
                edge_id = edge_indexer.id((('no_node_label',), bond))
                i_jedge_dict[i].append((j, edge_id))
        return nodes_fp, i_jedge_dict

    nodes = atoms
    i_jedge_dict = i_jbond_dict
    for _ in range(radius):
        # 1) 更新节点标签
        new_nodes = [None] * len(nodes)
        for i, j_edge in i_jedge_dict.items():
            neighbors = [(nodes[j], edge) for j, edge in j_edge]
            fingerprint = (nodes[i], tuple(sorted(neighbors)))
            new_nodes[i] = fp_indexer.id(fingerprint)
        nodes = new_nodes

        # 2) 基于新节点标签更新边标签
        _i_jedge_dict = defaultdict(lambda: [])
        for i, j_edge in i_jedge_dict.items():
            for j, edge in j_edge:
                both_side = tuple(sorted((nodes[i], nodes[j])))
                edge_id = edge_indexer.id((both_side, edge))
                _i_jedge_dict[i].append((j, edge_id))
        i_jedge_dict = _i_jedge_dict

    return np.array(nodes, dtype=np.int64), i_jedge_dict

def create_edge_arrays_from_dict(i_jedge_dict):
    """将 i -> [(j, edge_attr)] 的字典展开为有向边数组与属性数组"""
    edges, attrs = [], []
    for i, lst in i_jedge_dict.items():
        for j, eid in lst:
            edges.append([i, j])
            attrs.append(eid)
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.array(edges, dtype=np.int64), np.array(attrs, dtype=np.int64)

def create_bond_directed_edges(mol, bond_indexer):
    """按 RDKit 化学键生成有向边与‘键类型 id’属性"""
    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bid = bond_indexer.id(str(b.GetBondType()))
        edges.append([i, j]); attrs.append(bid)
        edges.append([j, i]); attrs.append(bid)
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.array(edges, dtype=np.int64), np.array(attrs, dtype=np.int64)

def create_adjacency(mol):
    return np.array(Chem.GetAdjacencyMatrix(mol), dtype=np.float32)

# =========================
# 处理一个 split
# =========================
def process_split(sequences, labels, atom_idx, bond_idx, edge_idx, fp_idx, radius):
    """
    返回：
      compounds: list[np.ndarray]    节点指纹 id，变长
      adjacencies: list[np.ndarray]  邻接矩阵，变形
      edges_pack: list[dict]         每个分子的边字典，含：
         - edge_index    : [E, 2] 有向边
         - edge_attr_wl  : [E]     WL 边指纹 id
         - edge_attr_bond: [E]     化学键类型 id
         - edge_index_bond: [E, 2] 化学键有向边
      labels_list: list[np.ndarray]  [[label], ...]
    """
    compounds, adjacencies, labels_list, edges_pack = [], [], [], []

    for k in range(len(sequences)):
        seq = sequences[k]
        lab = float(labels[k])

        mol = Chem.rdmolfiles.MolFromSequence(seq)
        if mol is None:
            raise ValueError(f'RDKit 无法从该序列构建分子: {seq}')

        # 节点与边标签迭代
        atoms = create_atoms(mol, atom_idx)
        i_jbond = create_ijbonddict(mol, bond_idx)
        nodes_fp, i_jedge_final = wl_update_nodes_edges(
            atoms, i_jbond, radius, fp_idx, edge_idx
        )
        edge_index_wl, edge_attr_wl = create_edge_arrays_from_dict(i_jedge_final)

        # 化学键有向边与“键类型 id”
        edge_index_bond, edge_attr_bond = create_bond_directed_edges(mol, bond_idx)

        # 邻接矩阵
        A = create_adjacency(mol)

        compounds.append(nodes_fp)
        adjacencies.append(A)
        labels_list.append(np.array([lab], dtype=np.float32))

        edges_pack.append(
            dict(
                edge_index=edge_index_wl,     # WL 边（与 edge_attr_wl 对齐）
                edge_attr_wl=edge_attr_wl,
                edge_attr_bond=edge_attr_bond,  # 与 edge_index_bond 对齐
                edge_index_bond=edge_index_bond
            )
        )

    return compounds, adjacencies, edges_pack, labels_list

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# =========================
# 主流程
# =========================
if __name__ == '__main__':
    # 读 CSV
    train_df = pd.read_csv('Data/train.csv')
    test_df  = pd.read_csv('Data/test.csv')
    indep_df = pd.read_csv('Data/independent_test.csv')

    train_seq = train_df['sequence'].values
    train_lab = train_df['label'].values
    test_seq  = test_df['sequence'].values
    test_lab  = test_df['label'].values
    indep_seq = indep_df['sequence'].values   # 修正：使用 independent_test.csv
    indep_lab = indep_df['label'].values      # 修正：使用 independent_test.csv

    # 1) 用 train 构建“可增长”的四类词典
    atom_dict = {}
    bond_dict = {}
    edge_dict = {}
    fp_dict_train_only = {}     # 仅由 train 数据构建的分子指纹词典

    atom_idx_train = GrowingIndexer(atom_dict)
    bond_idx_train = GrowingIndexer(bond_dict)
    edge_idx_train = GrowingIndexer(edge_dict)
    fp_idx_train   = GrowingIndexer(fp_dict_train_only)

    # 先跑一遍 train 以构建 train-only 词典并生成 train 编码
    tr_compounds, tr_adjs, tr_edges, tr_labels = process_split(
        train_seq, train_lab,
        atom_idx_train, bond_idx_train, edge_idx_train, fp_idx_train,
        RADIUS
    )

    # 2) 冻结词典 并补充 UNK（供推理编码使用，不影响 train-only 词典统计）
    if ATOM_UNK not in atom_dict: atom_dict[ATOM_UNK] = len(atom_dict)
    if BOND_UNK not in bond_dict: bond_dict[BOND_UNK] = len(bond_dict)
    if EDGE_UNK not in edge_dict: edge_dict[EDGE_UNK] = len(edge_dict)
    if FP_UNK   not in fp_dict_train_only: fp_dict_train_only[FP_UNK] = len(fp_dict_train_only)

    atom_idx_test  = FrozenIndexer(atom_dict, ATOM_UNK)
    bond_idx_test  = FrozenIndexer(bond_dict, BOND_UNK)
    edge_idx_test  = FrozenIndexer(edge_dict, EDGE_UNK)
    atom_idx_indep = FrozenIndexer(atom_dict, ATOM_UNK)
    bond_idx_indep = FrozenIndexer(bond_dict, BOND_UNK)
    edge_idx_indep = FrozenIndexer(edge_dict, EDGE_UNK)

    # —— 正式编码 test/independent：使用“冻结词典 + 基于 train 的 FP 映射 + UNK 回退”
    fp_idx_for_encode = FrozenIndexer(fp_dict_train_only, FP_UNK)
    te_compounds, te_adjs, te_edges, te_labels = process_split(
        test_seq, test_lab,
        atom_idx_test, bond_idx_test, edge_idx_test, fp_idx_for_encode,
        RADIUS
    )
    indep_compounds, indep_adjs, indep_edges, indep_labels = process_split(
        indep_seq, indep_lab,
        atom_idx_indep, bond_idx_indep, edge_idx_indep, fp_idx_for_encode,
        RADIUS
    )

    # 3) 额外：仅基于 test / independent 构建“测试指纹词典”（用于重叠统计，不参与推理编码）
    fp_dict_test_only = {}
    fp_idx_test_only = GrowingIndexer(fp_dict_test_only)
    _ = process_split(
        test_seq, test_lab,
        atom_idx_test, bond_idx_test, edge_idx_test, fp_idx_test_only,
        RADIUS
    )

    fp_dict_indep_only = {}
    fp_idx_indep_only = GrowingIndexer(fp_dict_indep_only)
    _ = process_split(
        indep_seq, indep_lab,
        atom_idx_indep, bond_idx_indep, edge_idx_indep, fp_idx_indep_only,
        RADIUS
    )

    # 4) 保存 —— 可变长对象用 pickle
    # 4.1 train
    save_pickle(tr_compounds, os.path.join(DIR_TRAIN, 'compounds.pkl'))
    save_pickle(tr_adjs,      os.path.join(DIR_TRAIN, 'adjacencies.pkl'))
    save_pickle(tr_edges,     os.path.join(DIR_TRAIN, 'edges.pkl'))
    tr_labels_arr = np.array(tr_labels, dtype=np.float32)
    np.save(os.path.join(DIR_TRAIN, 'labels.npy'), tr_labels_arr)

    # 4.2 test
    save_pickle(te_compounds, os.path.join(DIR_TEST, 'compounds.pkl'))
    save_pickle(te_adjs,      os.path.join(DIR_TEST,  'adjacencies.pkl'))
    save_pickle(te_edges,     os.path.join(DIR_TEST,  'edges.pkl'))
    te_labels_arr = np.array(te_labels, dtype=np.float32)
    np.save(os.path.join(DIR_TEST, 'labels.npy'), te_labels_arr)

    # 4.3 independent_test
    save_pickle(indep_compounds, os.path.join(DIR_INDEP, 'compounds.pkl'))
    save_pickle(indep_adjs,      os.path.join(DIR_INDEP, 'adjacencies.pkl'))
    save_pickle(indep_edges,     os.path.join(DIR_INDEP, 'edges.pkl'))
    indep_labels_arr = np.array(indep_labels, dtype=np.float32)
    np.save(os.path.join(DIR_INDEP, 'labels.npy'), indep_labels_arr)

    # 4.4 词典
    dump_dictionary(fp_dict_train_only, os.path.join(DIR_TRAIN, 'fingerprint_dict_from_train.pickle'))
    dump_dictionary(fp_dict_test_only,  os.path.join(DIR_TEST,  'fingerprint_dict_from_test.pickle'))
    dump_dictionary(fp_dict_indep_only, os.path.join(DIR_INDEP, 'fingerprint_dict_from_independent_test.pickle'))
    dump_dictionary(fp_dict_train_only, os.path.join(DIR_BASE,  'fingerprint_dict_from_train.pickle'))
    dump_dictionary(atom_dict, os.path.join(DIR_BASE, 'atom_dict.pickle'))
    dump_dictionary(bond_dict, os.path.join(DIR_BASE, 'bond_dict.pickle'))
    dump_dictionary(edge_dict, os.path.join(DIR_BASE, 'edge_dict.pickle'))

    # 4.5 快速统计
    def _len_stats(lengths):
        if len(lengths) == 0:
            return dict(min=0, mean=0, max=0)
        arr = np.array(lengths, dtype=np.int64)
        return dict(min=int(arr.min()),
                    mean=float(arr.mean()),
                    max=int(arr.max()))

    tr_lens = [len(x) for x in tr_compounds]
    te_lens = [len(x) for x in te_compounds]
    indep_lens = [len(x) for x in indep_compounds]

    # UNK 占比（基于用于编码 test/indep 的“train 词典”）
    unk_id = fp_idx_for_encode.unk_id
    te_tot_tokens = sum(len(x) for x in te_compounds)
    te_unk_tokens = sum(int(np.sum(x == unk_id)) for x in te_compounds) if te_tot_tokens > 0 else 0
    te_unk_ratio = (te_unk_tokens / te_tot_tokens) if te_tot_tokens > 0 else 0.0

    indep_tot_tokens = sum(len(x) for x in indep_compounds)
    indep_unk_tokens = sum(int(np.sum(x == unk_id)) for x in indep_compounds) if indep_tot_tokens > 0 else 0
    indep_unk_ratio = (indep_unk_tokens / indep_tot_tokens) if indep_tot_tokens > 0 else 0.0

    # 词典规模与重叠
    train_keys = set(fp_dict_train_only.keys()); train_keys.discard(FP_UNK)
    test_keys  = set(fp_dict_test_only.keys());  test_keys.discard(FP_UNK)
    indep_keys = set(fp_dict_indep_only.keys()); indep_keys.discard(FP_UNK)

    overlap_test  = train_keys & test_keys
    overlap_indep = train_keys & indep_keys

    meta_base = dict(
        radius=RADIUS,
        n_atom=len(atom_dict),
        n_bond=len(bond_dict),
        n_edge=len(edge_dict),
        n_fingerprint_train=len(train_keys),
        notes='推理编码使用“train 指纹词典 + UNK”；test/independent-only 词典仅用于统计覆盖率'
    )
    # train meta
    meta_train = dict(
        split='train',
        **meta_base,
        train_size=len(tr_compounds),
        train_fp_len_stats=_len_stats(tr_lens),
    )
    # test meta
    meta_test  = dict(
        split='test',
        **meta_base,
        n_fingerprint_test=len(test_keys),
        n_fingerprint_overlap=len(overlap_test),
        pct_test_in_train= round(len(overlap_test) / max(1, len(test_keys)), 6),
        pct_train_in_test= round(len(overlap_test) / max(1, len(train_keys)), 6),
        test_size=len(te_compounds),
        test_fp_len_stats=_len_stats(te_lens),
        test_unk_ratio_on_fp=round(te_unk_ratio, 6),
    )
    # independent meta
    meta_indep = dict(
        split='independent_test',
        **meta_base,
        n_fingerprint_independent=len(indep_keys),
        n_fingerprint_overlap=len(overlap_indep),
        pct_independent_in_train= round(len(overlap_indep) / max(1, len(indep_keys)), 6),
        pct_train_in_independent= round(len(overlap_indep) / max(1, len(train_keys)), 6),
        independent_size=len(indep_compounds),
        independent_fp_len_stats=_len_stats(indep_lens),
        independent_unk_ratio_on_fp=round(indep_unk_ratio, 6),
    )

    # 写 meta.json
    with open(os.path.join(DIR_BASE, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(all_splits=['train','test','independent_test'],
                       meta_train=meta_train, meta_test=meta_test, meta_independent=meta_indep),
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(DIR_TRAIN, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_train, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DIR_TEST,  'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_test,  f, ensure_ascii=False, indent=2)
    with open(os.path.join(DIR_INDEP, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_indep, f, ensure_ascii=False, indent=2)

    # 4.6 边属性 schema（便于下游读取）
    edge_schema = dict(
        edge_index_desc='有向边，形如 [[src, dst], ...]，与 edge_attr_wl 一一对应',
        edge_attr_wl_desc='WL 半径迭代后得到的“边指纹 id”，基于两端节点标签 + 键类型',
        edge_index_bond_desc='有向边，形如 [[src, dst], ...]，与 edge_attr_bond 一一对应',
        edge_attr_bond_desc='化学键类型 id（单、双、三、芳香等），来自 bond_dict.pickle'
    )
    for d in (DIR_TRAIN, DIR_TEST, DIR_INDEP):
        with open(os.path.join(d, 'edge_schema.json'), 'w', encoding='utf-8') as f:
            json.dump(edge_schema, f, ensure_ascii=False, indent=2)

    print('预处理完成：')
    print(json.dumps(dict(base=meta_base, train=meta_train, test=meta_test, independent=meta_indep),
                     ensure_ascii=False, indent=2))
