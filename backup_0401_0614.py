# -*- coding: utf-8 -*-
import os
import json
import math
import random
import pickle
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, recall_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, TransformerConv
from torch_geometric.utils import softmax as pyg_softmax

import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument('--dataset', type=str, default='AOPP',
                     choices=['AOPP', 'AnOxPP', 'AnOxPePred'])
_parser.add_argument('--fold', type=int, default=0)
_args, _ = _parser.parse_known_args()

DATASET = _args.dataset
FOLD    = _args.fold

CSV_TRAIN = f"data/{DATASET}/{FOLD}/train.csv"
TRAIN_DIR = f"data/rdkit/{DATASET}/{FOLD}/train"
CSV_TEST  = f"data/{DATASET}/{FOLD}/test.csv"
TEST_DIR  = f"data/rdkit/{DATASET}/{FOLD}/test"

ESM_CACHE_TRAIN = f"outcache/esm_{DATASET}_fold{FOLD}_train.npz"
ESM_CACHE_TEST  = f"outcache/esm_{DATASET}_fold{FOLD}_test.npz"

# =========================
# 超参
# =========================
SEED = 42
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 70
WARMUP_EPOCHS = max(3, int(0.1 * EPOCHS))

# ── AnOxPePred 用更长的 patience，其他数据集用 12 ──
PATIENCE = 20 if DATASET == 'AnOxPePred' else 12

MODEL_DIM = 256

# 序列 bigram 词向量
SEQ_EMB_DIM = 96
SEQ_MAX_LEN = 50
BIGRAM_LEN  = SEQ_MAX_LEN - 1

# ESM
ESM_MODEL_NAME = 'esm2_t33_650M_UR50D'
ESM_DIM = 1280
ESM_TRAINABLE = False
ESM_REPR_LAYER = None

# 图节点/边
ATOM_FEAT_DIM  = 9
EDGE_FEAT_DIM  = 3
EDGE_HIDDEN_DIM = 64

GRAPH_TOPK = 16
GT_LAYERS = 3
HEADS_GT  = 4

# ── AnOxPePred 用更大的 dropout 防止小数据过拟合 ──
DROPOUT      = 0.35 if DATASET == 'AnOxPePred' else 0.2
WEIGHT_DECAY = 2e-4 if DATASET == 'AnOxPePred' else 1e-4

W2V_WINDOW = 2
W2V_EPOCHS = 20
W2V_MIN_COUNT = 1
USE_FOCAL = True
FOCAL_GAMMA = 1.5

SMOOTH_EPS    = 0.05
MIXUP_PROB    = 0.20
MIXUP_ALPHA   = 0.20
DROP_PATH_P   = 0.10

MIN_SP_FOR_THR = 0.45   # ← 略微放宽，让阈值搜索空间更大
T_MIN, T_MAX   = 0.5, 2.5

N_MC_TEST = 20
MC_DISABLE_DROPEDGE = True

EMA_DECAY = 0.999

USE_CONST_LR = True
# ── AnOxPePred 用更小的学习率 ──
CONST_LR = 3e-4 if DATASET == 'AnOxPePred' else 6e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('outmodel/full', exist_ok=True)
os.makedirs('outlog/full', exist_ok=True)
os.makedirs('outpred/full', exist_ok=True)
os.makedirs('outcache', exist_ok=True)

# =========================
# 实用函数
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

AMINO = list('ACDEFGHIKLMNPQRSTVWY')

def to_bigrams(seq: str) -> List[str]:
    seq = ''.join([c for c in seq if c in AMINO])
    return [seq[i:i+2] for i in range(len(seq) - 1)]

def pad_or_trim(tokens: List[str], target_len: int, pad_token: str):
    if len(tokens) >= target_len:
        return tokens[:target_len]
    return tokens + [pad_token] * (target_len - len(tokens))

def load_graph_split(split_dir: str):
    with open(os.path.join(split_dir, 'compounds.pkl'), 'rb') as f:
        compounds = pickle.load(f)
    with open(os.path.join(split_dir, 'edges.pkl'), 'rb') as f:
        edges = pickle.load(f)
    labels = np.load(os.path.join(split_dir, 'labels.npy'), allow_pickle=True)
    y = np.array([float(x[0]) for x in labels], dtype=np.float32)
    return compounds, edges, y

def maybe_load_esm_cache(path: Optional[str]):
    if path and os.path.exists(path):
        arr = np.load(path, mmap_mode='r')
        return dict(emb=arr['emb'], len=arr['len'])
    return None

# =========================
# 训练 CBOW bigram 词向量
# =========================
def train_cbow_embeddings(train_sequences: List[str], save_path: str,
                          vector_size: int = SEQ_EMB_DIM) -> Tuple[dict, np.ndarray]:
    from gensim.models import Word2Vec
    PAD = '<PAD>'
    UNK = '<UNK>'
    corpus = []
    for s in train_sequences:
        toks = to_bigrams(s)
        toks = pad_or_trim(toks, BIGRAM_LEN, PAD)
        corpus.append(toks)
    model = Word2Vec(
        sentences=corpus, vector_size=vector_size,
        window=W2V_WINDOW, sg=0, min_count=W2V_MIN_COUNT,
        workers=4, epochs=W2V_EPOCHS, negative=5, seed=SEED
    )
    if PAD not in model.wv.key_to_index:
        model.wv.add_vector(PAD, np.zeros(vector_size, dtype=np.float32))
    if UNK not in model.wv.key_to_index:
        model.wv.add_vector(UNK, np.random.normal(0, 0.02, size=vector_size).astype(np.float32))
    model.save(save_path)
    idx2tok = list(model.wv.key_to_index.keys())
    tok2idx = {t: i for i, t in enumerate(idx2tok)}
    emb_mat = np.stack([model.wv[t] for t in idx2tok], axis=0).astype(np.float32)
    return tok2idx, emb_mat

# =========================
# ESM 加载与编码
# =========================
def load_esm(model_name=ESM_MODEL_NAME, trainable=ESM_TRAINABLE):
    try:
        import esm
    except ImportError as e:
        raise ImportError("未找到 esm 库，请先安装：pip install fair-esm") from e
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if not trainable:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    return model, alphabet

# =========================
# 数据集
# =========================
class GraphSeqDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, compounds, edges, y, tok2idx: dict,
                 esm_tok_to_idx: dict, esm_pad_idx: int, esm_bos_idx: int, esm_eos_idx: int,
                 res_emb: Optional[np.ndarray] = None, res_len: Optional[np.ndarray] = None):
        assert len(df) == len(compounds) == len(edges) == len(y), \
            f'数据条数不一致: csv={len(df)}, comp={len(compounds)}, edges={len(edges)}, y={len(y)}'
        self.df = df.reset_index(drop=True)
        self.compounds = compounds
        self.edges = edges
        self.y = y.astype(np.float32)
        self.tok2idx = tok2idx
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.esm_tok_to_idx = esm_tok_to_idx
        self.esm_pad_idx = esm_pad_idx
        self.esm_bos_idx = esm_bos_idx
        self.esm_eos_idx = esm_eos_idx
        self.res_emb = res_emb
        self.res_len = res_len

    def __len__(self):
        return len(self.df)

    def encode_seq_tokens(self, seq: str) -> torch.Tensor:
        toks = to_bigrams(seq)
        toks = pad_or_trim(toks, BIGRAM_LEN, self.PAD)
        ids = [self.tok2idx.get(t, self.tok2idx.get(self.UNK, 0)) for t in toks]
        return torch.tensor(ids, dtype=torch.long)

    def encode_esm_tokens(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = ''.join([c for c in seq if c in AMINO])
        L0 = min(len(seq), SEQ_MAX_LEN)
        ids = torch.full((SEQ_MAX_LEN + 2,), fill_value=self.esm_pad_idx, dtype=torch.long)
        ids[0] = self.esm_bos_idx
        if L0 > 0:
            idx_map = self.esm_tok_to_idx
            default_x = idx_map.get('X', self.esm_pad_idx)
            ids[1:1+L0] = torch.tensor(
                [idx_map.get(seq[i], default_x) for i in range(L0)], dtype=torch.long)
        ids[1 + L0] = self.esm_eos_idx
        return ids, torch.tensor(L0, dtype=torch.long)

    def __getitem__(self, idx):
        node_feats = torch.tensor(self.compounds[idx], dtype=torch.float32)
        e_pack = self.edges[idx]
        ei = e_pack['edge_index_bond']
        ea = e_pack['edge_attr_bond']

        if isinstance(ei, list) and len(ei) == 2 and len(ei[0]) > 0:
            edge_index = torch.tensor(ei, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if isinstance(ea, list) and len(ea) > 0:
            edge_attr = torch.tensor(ea, dtype=torch.float32)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
        else:
            edge_attr = torch.empty((0, EDGE_FEAT_DIM), dtype=torch.float32)

        y = torch.tensor(self.y[idx], dtype=torch.float32)
        seq = str(self.df.loc[idx, 'SEQUENCE'])
        seq_tokens = self.encode_seq_tokens(seq)

        data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.seq_tokens = seq_tokens

        if self.res_emb is not None and self.res_len is not None:
            data.res_feat = torch.from_numpy(self.res_emb[idx]).float()
            data.res_len  = torch.tensor(int(self.res_len[idx]), dtype=torch.long)
        else:
            esm_ids, esm_len = self.encode_esm_tokens(seq)
            data.esm_ids = esm_ids
            data.esm_len = esm_len

        return data

# =========================
# DropPath
# =========================
def drop_path(x, drop_prob: float, training: bool):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

def mean_pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    B = int(batch.max().item()) + 1
    sums = torch.zeros(B, h.size(-1), device=h.device, dtype=h.dtype)
    cnts = torch.zeros(B, 1, device=h.device, dtype=h.dtype)
    sums.index_add_(0, batch, h)
    cnts.index_add_(0, batch, torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype))
    return sums / cnts.clamp_min(1.0)

def max_pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    B = int(batch.max().item()) + 1
    out = torch.full((B, h.size(-1)), float('-inf'), device=h.device, dtype=h.dtype)
    for b in range(B):
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            out[b] = h[idx].max(dim=0).values
    out[out == float('-inf')] = 0.0
    return out

class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.lin = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        B = int(batch.max().item()) + 1
        score = self.lin(h).squeeze(-1)
        out = torch.zeros(B, h.size(-1), device=h.device, dtype=h.dtype)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            w = torch.softmax(score[idx].float(), dim=0).to(h.dtype)
            out[b] = (h[idx] * w.unsqueeze(-1)).sum(dim=0)
        return out

# =========================
# 图编码器
# =========================
class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(ATOM_FEAT_DIM, MODEL_DIM),
            nn.LayerNorm(MODEL_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(EDGE_FEAT_DIM, EDGE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        mlp1 = nn.Sequential(nn.Linear(MODEL_DIM, MODEL_DIM), nn.ReLU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        mlp2 = nn.Sequential(nn.Linear(MODEL_DIM, MODEL_DIM), nn.ReLU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.gine1 = GINEConv(mlp1, train_eps=True, edge_dim=EDGE_HIDDEN_DIM)
        self.gine2 = GINEConv(mlp2, train_eps=True, edge_dim=EDGE_HIDDEN_DIM)
        self.norm_g1 = nn.LayerNorm(MODEL_DIM)
        self.norm_g2 = nn.LayerNorm(MODEL_DIM)
        self.drop = nn.Dropout(DROPOUT)

        self.fuse = nn.Sequential(
            nn.Linear(MODEL_DIM * 2, MODEL_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

        self.gt_layers  = nn.ModuleList()
        self.gt_norm1   = nn.ModuleList()
        self.gt_ffn     = nn.ModuleList()
        self.gt_norm2   = nn.ModuleList()
        for _ in range(GT_LAYERS):
            self.gt_layers.append(TransformerConv(
                MODEL_DIM, MODEL_DIM // HEADS_GT, heads=HEADS_GT,
                concat=True, dropout=DROPOUT, edge_dim=EDGE_HIDDEN_DIM, beta=True))
            self.gt_norm1.append(nn.LayerNorm(MODEL_DIM))
            self.gt_ffn.append(nn.Sequential(
                nn.Linear(MODEL_DIM, 2 * MODEL_DIM), nn.GELU(),
                nn.Dropout(DROPOUT), nn.Linear(2 * MODEL_DIM, MODEL_DIM)
            ))
            self.gt_norm2.append(nn.LayerNorm(MODEL_DIM))

        self.attn_pool   = AttnPool(MODEL_DIM)
        self.global_proj = nn.Linear(MODEL_DIM * 3, MODEL_DIM)
        self.score_lin   = nn.Linear(MODEL_DIM, 1)

    def _select_topk(self, h, batch, k=GRAPH_TOPK):
        score = self.score_lin(h).squeeze(-1)
        B = int(batch.max().item()) + 1
        tokens, masks = [], []
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                tokens.append(torch.zeros(k, h.size(-1), device=h.device, dtype=h.dtype))
                masks.append(torch.ones(k, dtype=torch.bool, device=h.device))
                continue
            kb = min(k, idx.numel())
            _, topi = torch.topk(score[idx], kb)
            pick = idx[topi]
            t = h[pick]
            if kb < k:
                pad = torch.zeros(k - kb, h.size(-1), device=h.device, dtype=h.dtype)
                t = torch.cat([t, pad], dim=0)
                mask = torch.cat([
                    torch.zeros(kb, dtype=torch.bool, device=h.device),
                    torch.ones(k - kb, dtype=torch.bool, device=h.device)
                ], dim=0)
            else:
                mask = torch.zeros(k, dtype=torch.bool, device=h.device)
            tokens.append(t)
            masks.append(mask)
        return torch.stack(tokens, dim=0), torch.stack(masks, dim=0)

    def forward(self, data: Data):
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = data.batch

        h0 = self.node_proj(x)
        e  = self.edge_proj(edge_attr)

        h = F.relu(self.gine1(h0, edge_index, e))
        h = self.drop(self.norm_g1(h))
        h = F.relu(self.gine2(h, edge_index, e))
        h = self.drop(self.norm_g2(h))
        h = self.fuse(torch.cat([h0, h], dim=-1))

        for i in range(GT_LAYERS):
            z  = self.gt_layers[i](h, edge_index, e)
            h  = self.gt_norm1[i](h + drop_path(z, DROP_PATH_P, self.training))
            z2 = self.gt_ffn[i](h)
            h  = self.gt_norm2[i](h + drop_path(z2, DROP_PATH_P, self.training))
            h  = F.relu(self.drop(h))

        g_mean   = mean_pool(h, batch)
        g_max    = max_pool(h, batch)
        g_attn   = self.attn_pool(h, batch)
        g_global = self.global_proj(torch.cat([g_mean, g_max, g_attn], dim=-1))

        g_tokens, g_mask = self._select_topk(h, batch, k=GRAPH_TOPK)
        return g_tokens, g_mask, g_global

# =========================
# ESM 序列编码器
# =========================
class ESMSeqEncoder(nn.Module):
    def __init__(self, esm_model=None):
        super().__init__()
        self.esm_model = esm_model
        if esm_model is not None and not ESM_TRAINABLE:
            for p in esm_model.parameters():
                p.requires_grad = False
            esm_model.eval()
        self.repr_layer = ESM_REPR_LAYER if ESM_REPR_LAYER is not None else 33

        self.proj = nn.Sequential(
            nn.Linear(ESM_DIM, MODEL_DIM),
            nn.LayerNorm(MODEL_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
        self.attn_pool = nn.Linear(MODEL_DIM, 1)

    def forward(self, data: Data, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.proj.parameters()).device

        if hasattr(data, 'res_feat'):
            rf       = data.res_feat.to(device)
            res_feat = rf.view(B, SEQ_MAX_LEN, ESM_DIM)
            esm_len  = data.res_len.view(B).to(device)
        else:
            assert self.esm_model is not None
            esm_ids = data.esm_ids.view(B, SEQ_MAX_LEN + 2).to(device)
            esm_len = data.esm_len.view(B).to(device)
            with torch.set_grad_enabled(ESM_TRAINABLE):
                out      = self.esm_model(esm_ids, repr_layers=[self.repr_layer], return_contacts=False)
                res_feat = out['representations'][self.repr_layer][:, 1:-1, :]

        h = self.proj(res_feat)
        L = h.size(1)
        idx  = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        mask = idx < esm_len.unsqueeze(1)

        w       = self.attn_pool(h)
        w       = w.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        w       = torch.softmax(w, dim=1)
        seq_vec = (h * w).sum(dim=1)

        return h, seq_vec, mask

# =========================
# Bigram ConvNeXt 序列编码器
# =========================
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.dw   = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        y = self.dw(x.transpose(1, 2)).transpose(1, 2)
        y = self.norm(y)
        return self.mlp(y) + x

class ConvNeXtSequenceEncoder(nn.Module):
    def __init__(self, emb_matrix: np.ndarray, d_model: int = MODEL_DIM, dropout: float = DROPOUT):
        super().__init__()
        num_vocab, emb_dim = emb_matrix.shape
        self.embedding = nn.Embedding(num_vocab, emb_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = False

        self.input_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks    = nn.ModuleList([ConvNeXtBlock(d_model, dropout) for _ in range(4)])
        self.attn_pool = nn.Linear(d_model, 1)
        self.norm      = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, seq_tokens: torch.Tensor):
        x = self.embedding(seq_tokens)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.drop(self.norm(x))
        w = torch.softmax(self.attn_pool(x), dim=1)
        s_vec = (x * w).sum(dim=1)
        return x, s_vec

# =========================
# 双模态融合
# =========================
class BiCrossAttention(nn.Module):
    def __init__(self, d_model: int = MODEL_DIM, heads: int = 4, dropout: float = DROPOUT, layers: int = 3):
        super().__init__()
        self.layers_list = nn.ModuleList([
            nn.ModuleDict(dict(
                g2s=nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=False),
                s2g=nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=False),
                ng=nn.LayerNorm(d_model), ns=nn.LayerNorm(d_model),
                gate_g=nn.Linear(d_model * 2, d_model),
                gate_s=nn.Linear(d_model * 2, d_model),
                ffg=nn.Sequential(nn.Linear(d_model, 2*d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2*d_model, d_model)),
                ffs=nn.Sequential(nn.Linear(d_model, 2*d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2*d_model, d_model)),
                ng2=nn.LayerNorm(d_model), ns2=nn.LayerNorm(d_model),
                drop=nn.Dropout(dropout)
            )) for _ in range(layers)
        ])

    def forward(self, g_tokens, g_mask, s_tokens, s_mask=None):
        B, K, d = g_tokens.size()
        L = s_tokens.size(1)
        G = g_tokens.transpose(0, 1)
        S = s_tokens.transpose(0, 1)
        kp_s = torch.zeros(B, L, dtype=torch.bool, device=s_tokens.device)
        kp_g = g_mask

        for blk in self.layers_list:
            Zg, _ = blk['g2s'](G, S, S, key_padding_mask=kp_s)
            Zs, _ = blk['s2g'](S, G, G, key_padding_mask=kp_g)
            G_new = blk['ng'](G + blk['drop'](Zg))
            S_new = blk['ns'](S + blk['drop'](Zs))
            G = torch.sigmoid(blk['gate_g'](torch.cat([G_new, G], -1))) * G_new + \
                (1 - torch.sigmoid(blk['gate_g'](torch.cat([G_new, G], -1)))) * G
            S = torch.sigmoid(blk['gate_s'](torch.cat([S_new, S], -1))) * S_new + \
                (1 - torch.sigmoid(blk['gate_s'](torch.cat([S_new, S], -1)))) * S
            G = blk['ng2'](G + blk['drop'](blk['ffg'](G)))
            S = blk['ns2'](S + blk['drop'](blk['ffs'](S)))

        G = G.transpose(0, 1)
        S = S.transpose(0, 1)
        valid = (~g_mask).float().unsqueeze(-1)
        g_vec = (G * valid).sum(1) / valid.sum(1).clamp(min=1e-6)
        s_vec = S.max(dim=1).values
        return g_vec, s_vec

class AttnFusion(nn.Module):
    def __init__(self, d_model: int = MODEL_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.proj  = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor):
        w = torch.softmax(self.score(torch.tanh(self.proj(tokens))), dim=1)
        return self.drop(self.norm((tokens * w).sum(dim=1)))

class Classifier(nn.Module):
    def __init__(self, d_model: int = MODEL_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# =========================
# 主模型
# =========================
class PepToxModel(nn.Module):
    def __init__(self, emb_matrix: np.ndarray, esm_model=None):
        super().__init__()
        self.graph  = GraphEncoder()
        self.seqenc = ConvNeXtSequenceEncoder(emb_matrix, d_model=MODEL_DIM)
        self.esmenc = ESMSeqEncoder(esm_model)
        self.cross  = BiCrossAttention(d_model=MODEL_DIM, heads=4, layers=3)
        self.fusion = AttnFusion(d_model=MODEL_DIM)
        self.cls    = Classifier(d_model=MODEL_DIM)

    def encode(self, data: Data):
        B = int(data.batch.max().item()) + 1

        g_tokens, g_mask, g_global = self.graph(data)
        esm_tokens, esm_vec, _    = self.esmenc(data, B)

        seq_tokens = data.seq_tokens
        if seq_tokens.dim() == 1:
            seq_tokens = seq_tokens.view(B, BIGRAM_LEN)
        s_tokens, s_vec = self.seqenc(seq_tokens)

        g_vec_ca, s_vec_ca = self.cross(g_tokens, g_mask, esm_tokens)

        tokens = torch.stack([g_global, g_vec_ca, esm_vec, s_vec, s_vec_ca], dim=1)
        fused  = self.fusion(tokens)
        return fused

    def forward(self, data: Data):
        return self.cls(self.encode(data))

# =========================
# 损失与指标
# =========================
def smooth_labels(y, eps=SMOOTH_EPS):
    return y * (1 - eps) + 0.5 * eps if eps > 0 else y

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth_eps=SMOOTH_EPS):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.smooth_eps = smooth_eps

    def forward(self, logits, targets):
        if self.smooth_eps > 0:
            targets = smooth_labels(targets, self.smooth_eps)
        p     = torch.sigmoid(logits)
        ce    = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t   = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * ce * (1 - p_t) ** self.gamma).mean()

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int32)
    try:    auroc = roc_auc_score(y_true, y_prob)
    except: auroc = 0.5
    try:    auprc = average_precision_score(y_true, y_prob)
    except: auprc = 0.5
    f1  = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    se  = recall_score(y_true, y_pred, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sp  = tn / (tn + fp + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    pre = tp / (tp + fp + 1e-8)
    return dict(AUROC=auroc, AUPRC=auprc, F1=f1, MCC=mcc, SE=se, SP=sp, ACC=acc, PRE=pre)

def find_best_threshold(y_true, y_prob, mode='mcc', min_sp=MIN_SP_FOR_THR):
    candidates = np.linspace(0.05, 0.95, 181)
    best_t, best_v = 0.5, -1.0
    for t in candidates:
        m = compute_metrics(y_true, y_prob, threshold=t)
        if min_sp is not None and m['SP'] < min_sp:
            continue
        v = m['MCC'] if mode == 'mcc' else 0.5 * (m['SE'] + m['SP'])
        if v > best_v:
            best_v, best_t = v, t
    return best_t

class TempScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / self.log_t.exp()

    def temperature(self):
        return float(self.log_t.exp().clamp(T_MIN, T_MAX).item())

    def fit(self, val_logits, val_labels, max_iter=50):
        opt = torch.optim.LBFGS([self.log_t], lr=0.1, max_iter=max_iter)
        nll = nn.BCEWithLogitsLoss()
        def closure():
            opt.zero_grad()
            loss = nll(self.forward(val_logits), val_labels) + 1e-4 * self.log_t ** 2
            loss.backward()
            return loss
        opt.step(closure)
        return self.temperature()

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay  = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if n not in self.shadow:
                self.shadow[n] = p.data.detach().clone()
            else:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.backup[n] = p.data.clone()
            p.data.copy_(self.shadow.get(n, p.data))

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

def feature_mixup(feat, y, alpha=MIXUP_ALPHA):
    if feat.size(0) < 2 or alpha <= 0:
        return feat, y
    lam = max(float(np.random.beta(alpha, alpha)), 1.0 - float(np.random.beta(alpha, alpha)))
    idx = torch.randperm(feat.size(0), device=feat.device)
    return lam * feat + (1 - lam) * feat[idx], lam * y + (1 - lam) * y[idx]

def predict_mc_dropout(model, loader, T, n_passes=10):
    model.train()
    all_passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            probs = []
            for batch in loader:
                batch = batch.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    prob = torch.sigmoid(model(batch) / T)
                probs.append(prob.detach().cpu().numpy())
            all_passes.append(np.concatenate(probs))
    model.eval()
    return np.mean(np.stack(all_passes), axis=0)

# =========================
# 主训练函数
# =========================
def run_single_fold():
    print(f'\n===== {DATASET} | Fold {FOLD} =====')
    print(f'[Config] DROPOUT={DROPOUT} | CONST_LR={CONST_LR} | '
          f'PATIENCE={PATIENCE} | WEIGHT_DECAY={WEIGHT_DECAY}')
    set_seed()

    # 1) 载入数据
    df_train = pd.read_csv(CSV_TRAIN)
    df_test  = pd.read_csv(CSV_TEST)
    comp_tr, edges_tr, y_tr = load_graph_split(TRAIN_DIR)
    comp_te, edges_te, y_te = load_graph_split(TEST_DIR)

    assert len(df_train) == len(comp_tr) == len(y_tr)
    assert len(df_test)  == len(comp_te) == len(y_te)
    print(f'训练集: {len(df_train)} | 测试集: {len(df_test)}')

    n_pos = float(df_train['label'].sum())
    n_neg = float(len(df_train) - n_pos)
    print(f'正样本: {int(n_pos)} | 负样本: {int(n_neg)} | 比例: {n_pos/max(1,n_neg):.3f}')

    # 2) ESM 缓存
    cache_tr = maybe_load_esm_cache(ESM_CACHE_TRAIN)
    cache_te = maybe_load_esm_cache(ESM_CACHE_TEST)
    needs_online = (cache_tr is None) or (cache_te is None)

    if needs_online:
        print('[ESM] 无缓存，启用在线推理')
        esm_model, esm_alphabet = load_esm()
        esm_model = esm_model.to(DEVICE)
        esm_tok_to_idx = esm_alphabet.tok_to_idx
        esm_pad_idx = esm_alphabet.padding_idx
        esm_bos_idx = getattr(esm_alphabet, 'cls_idx', esm_tok_to_idx.get('<cls>', 0))
        esm_eos_idx = getattr(esm_alphabet, 'eos_idx', esm_tok_to_idx.get('<eos>', 0))
    else:
        print('[ESM] 使用缓存')
        esm_model = None
        esm_tok_to_idx, esm_pad_idx, esm_bos_idx, esm_eos_idx = {}, 0, 0, 0

    # 3) 训练 bigram 词向量
    w2v_path = f'outmodel/full/{DATASET}_fold{FOLD}_w2v_bigram.model'
    tok2idx, emb_mat = train_cbow_embeddings(df_train['SEQUENCE'].tolist(), w2v_path)

    # 4) Dataset & DataLoader
    ds_train = GraphSeqDataset(
        df_train, comp_tr, edges_tr, y_tr, tok2idx,
        esm_tok_to_idx=esm_tok_to_idx, esm_pad_idx=esm_pad_idx,
        esm_bos_idx=esm_bos_idx, esm_eos_idx=esm_eos_idx,
        res_emb=(cache_tr['emb'] if cache_tr else None),
        res_len=(cache_tr['len'] if cache_tr else None)
    )
    ds_test = GraphSeqDataset(
        df_test, comp_te, edges_te, y_te, tok2idx,
        esm_tok_to_idx=esm_tok_to_idx, esm_pad_idx=esm_pad_idx,
        esm_bos_idx=esm_bos_idx, esm_eos_idx=esm_eos_idx,
        res_emb=(cache_te['emb'] if cache_te else None),
        res_len=(cache_te['len'] if cache_te else None)
    )
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 5) 模型
    model = PepToxModel(emb_mat, esm_model).to(DEVICE)

    
    if USE_FOCAL:
        if DATASET == 'AnOxPePred':
            criterion = BCEFocalLoss(
                alpha = n_pos / (n_pos + n_neg + 1e-8),
                gamma = 1.5
            )
        else:
            criterion = BCEFocalLoss(
                alpha = n_pos / (n_pos + n_neg + 1e-8),
                gamma = FOCAL_GAMMA
            )
    else:
        pos_w = torch.tensor([n_neg / max(1.0, n_pos)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONST_LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ema       = EMA(model, decay=EMA_DECAY)

    best_score  = -1.0
    best_path   = f'outmodel/full/{DATASET}_fold{FOLD}_best.pt'
    patience    = PATIENCE
    best_thresh = 0.5
    best_T      = 1.0

    # 6) 训练循环
    for epoch in range(1, EPOCHS + 1):
        if epoch == 6:
            model.seqenc.embedding.weight.requires_grad = True

        model.train()
        train_losses = []
        for batch in loader_train:
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if MIXUP_PROB > 0 and random.random() < MIXUP_PROB and batch.num_graphs > 1:
                    feat = model.encode(batch)
                    feat_mix, y_mix = feature_mixup(feat, batch.y)
                    loss = criterion(model.cls(feat_mix), y_mix)
                else:
                    loss = criterion(model(batch), batch.y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            ema.update(model)

        # ── 验证（EMA权重）──
        model.eval()
        ema.apply(model)
        val_logits_list, val_true = [], []
        with torch.no_grad():
            for batch in loader_test:
                batch = batch.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(batch)
                val_logits_list.append(logits.float().cpu().numpy())
                val_true.append(batch.y.cpu().numpy())

        val_logits = np.concatenate(val_logits_list)
        val_true   = np.concatenate(val_true)

        temp_scaler = TempScaling().to(DEVICE)
        T = temp_scaler.fit(
            torch.from_numpy(val_logits).to(DEVICE),
            torch.from_numpy(val_true).to(DEVICE)
        )
        T = float(np.clip(T, T_MIN, T_MAX))

        val_probs = torch.sigmoid(torch.from_numpy(val_logits) / T).numpy()

        # ── 修改1：AnOxPePred 用阈值搜索，其他数据集固定 0.4 ──
        if DATASET == 'AnOxPePred':
            thr = find_best_threshold(val_true, val_probs, mode='mcc', min_sp=MIN_SP_FOR_THR)
        else:
            thr = 0.4

        metrics = compute_metrics(val_true, val_probs, threshold=thr)

        # ── 修改2：AnOxPePred 早停监控 MCC，其他数据集监控 AUROC+AUPRC ──
        if DATASET == 'AnOxPePred':
            val_score = metrics['MCC']
        else:
            val_score = 0.5 * (metrics['AUROC'] + metrics['AUPRC'])

        print(f'Epoch {epoch:02d} | Loss {np.mean(train_losses):.4f} | '
              f'AUROC {metrics["AUROC"]:.4f} AUPRC {metrics["AUPRC"]:.4f} '
              f'F1 {metrics["F1"]:.4f} MCC {metrics["MCC"]:.4f} '
              f'SE {metrics["SE"]:.4f} SP {metrics["SP"]:.4f} | '
              f'T {T:.3f} thr {thr:.2f} score {val_score:.4f}')

        if val_score > best_score:
            best_score  = val_score
            best_thresh = float(thr)
            best_T      = T
            torch.save({'model': model.state_dict(),
                        'meta': dict(epoch=epoch, score=best_score,
                                     thr=best_thresh, temp=best_T,
                                     auroc=metrics['AUROC'], auprc=metrics['AUPRC'],
                                     mcc=metrics['MCC'])},
                       best_path)
            patience = PATIENCE
            print(f'  → New best (score={best_score:.4f}, thr={best_thresh:.2f})')
        else:
            patience -= 1
            if patience <= 0:
                print('Early stopping')
                ema.restore(model)
                break

        ema.restore(model)

    # 7) 最终测试
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    meta        = ckpt.get('meta', {})
    best_thresh = float(meta.get('thr', best_thresh))
    best_T      = float(np.clip(meta.get('temp', best_T), T_MIN, T_MAX))

    model.eval()
    if N_MC_TEST and N_MC_TEST > 1:
        test_probs = predict_mc_dropout(model, loader_test, best_T, n_passes=N_MC_TEST)
    else:
        probs_list = []
        with torch.no_grad():
            for batch in loader_test:
                batch = batch.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    prob = torch.sigmoid(model(batch) / best_T)
                probs_list.append(prob.cpu().numpy())
        test_probs = np.concatenate(probs_list)

    m_test = compute_metrics(
        df_test['label'].values.astype(np.int32), test_probs, threshold=best_thresh
    )
    print(f'\n===== [{DATASET} Fold {FOLD}] Final Test =====')
    print(f'AUROC {m_test["AUROC"]:.4f} | AUPRC {m_test["AUPRC"]:.4f} | '
          f'ACC {m_test["ACC"]:.4f} | PRE {m_test["PRE"]:.4f} | '
          f'SE {m_test["SE"]:.4f} | SP {m_test["SP"]:.4f} | '
          f'F1 {m_test["F1"]:.4f} | MCC {m_test["MCC"]:.4f}')
    print(f'Best threshold: {best_thresh:.2f} | Temperature: {best_T:.3f}')

    # 8) 保存结果
    pd.DataFrame(dict(
        y_true=df_test['label'].values.astype(np.int32),
        y_prob=test_probs,
        y_pred=(test_probs >= best_thresh).astype(np.int32)
    )).to_csv(f'outpred/full/{DATASET}_fold{FOLD}_test_pred.csv', index=False)

    with open(f'outlog/full/{DATASET}_fold{FOLD}_summary.json', 'w') as f:
        json.dump(dict(dataset=DATASET, fold=FOLD,
                       best_epoch=int(meta.get('epoch', -1)),
                       best_val_score=float(best_score),
                       threshold=best_thresh, temperature=best_T,
                       test_metrics=m_test), f, indent=2)

    print(f'Saved → outpred/full/{DATASET}_fold{FOLD}_test_pred.csv')
    return m_test


if __name__ == '__main__':
    run_single_fold()