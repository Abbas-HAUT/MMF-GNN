#!/usr/bin/env python3
"""
MMF-GNN: Multi-Modal Fusion Graph Neural Network for Drug Discovery Activity

A novel architecture for predicting anti-schistosome activity of small molecules.

Usage:
    python train.py --data_path /path/to/data1.csv /path/to/data2.csv --output_dir ./output --epochs 100

Author: Abbas
"""

import os
import sys
import argparse
import math
import warnings
from datetime import datetime

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, average_precision_score,
                             cohen_kappa_score, matthews_corrcoef,
                             classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve, auc)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Configuration and Constants
# ============================================================
class Config:
    """Configuration class for MMF-GNN."""

    SMILES_DICT = {
        "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
        "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
        "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
        "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
        "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
        "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
        "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
        "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64,
        "*": 65
    }

    ATOM_LIST = ['C', 'H', 'O', 'N', 'P', 'S', 'Cl', 'I', 'Mg', 'Se', 'F', 'As',
                 'Fe', 'Na', 'Br', 'Cu', 'Os', 'Co', 'Mo', 'R', '*', 'Hg', 'Au',
                 'Sb', 'Si', 'B', 'Cd', 'Pt', 'Ca']

    BOND_TYPES = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    MAX_SMILES_LEN = 200
    NUM_CHARS = 65
    NODE_FEATURES = 42
    EDGE_FEATURES = 10


# ============================================================
# Feature Extraction Functions
# ============================================================
def get_atom_features(atom):
    """Enhanced atom features (42 dimensions)."""
    atom_type = [0] * len(Config.ATOM_LIST)
    symbol = atom.GetSymbol()
    if symbol in Config.ATOM_LIST:
        atom_type[Config.ATOM_LIST.index(symbol)] = 1
    else:
        atom_type[Config.ATOM_LIST.index('*')] = 1

    hybridization = [0, 0, 0, 0, 0]
    hyb = atom.GetHybridization()
    hyb_types = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
    if hyb in hyb_types:
        hybridization[hyb_types.index(hyb)] = 1

    properties = [
        atom.GetFormalCharge(),
        atom.GetDegree(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetMass() / 100.0,
        int(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    ]

    return atom_type + hybridization + properties


def get_bond_features(bond):
    """Bond/Edge features (10 dimensions)."""
    bond_type = [0, 0, 0, 0]
    bt = bond.GetBondType()
    if bt in Config.BOND_TYPES:
        bond_type[Config.BOND_TYPES[bt]] = 1

    features = [
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
        bond.GetBondTypeAsDouble() / 3.0,
        int(bond.GetIsAromatic()),
        1.0
    ]

    return bond_type + features


def smiles_to_enhanced_graph(smiles):
    """Convert SMILES to enhanced molecular graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = []
    edge_features = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_features.extend([bond_feat, bond_feat])

    if not edge_index:
        edge_index = [[0, 0]]
        edge_features = [[0] * Config.EDGE_FEATURES]

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return x, edge_index, edge_attr


def smiles_to_label_encoding(smiles, max_len=Config.MAX_SMILES_LEN):
    """Label encoding for SMILES."""
    toks = [Config.SMILES_DICT.get(char, 0) for char in smiles]
    if len(toks) > max_len:
        toks = toks[:max_len]
    else:
        toks = toks + [0] * (max_len - len(toks))
    return toks


def smiles_to_onehot_encoding(smiles, max_len=Config.MAX_SMILES_LEN):
    """One-hot encoding for SMILES."""
    one_hot = np.zeros((max_len, Config.NUM_CHARS), dtype=np.float32)
    for i, char in enumerate(smiles[:max_len]):
        if char in Config.SMILES_DICT:
            one_hot[i, Config.SMILES_DICT[char] - 1] = 1
    return one_hot


# ============================================================
# Model Components
# ============================================================
class MultiScaleGraphAttention(nn.Module):
    """Multi-Scale Graph Attention (MSGA)."""
    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.scale1 = GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.scale2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.scale3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm3 = nn.LayerNorm(hidden_dim * num_heads)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, edge_index):
        h1 = self.activation(self.dropout(self.norm1(self.scale1(x, edge_index))))
        h2 = self.activation(self.dropout(self.norm2(self.scale2(h1, edge_index))))
        h3 = self.activation(self.norm3(self.scale3(h2, edge_index)))
        weights = F.softmax(self.scale_weights, dim=0)
        return weights[0] * h1 + weights[1] * h2 + weights[2] * h3


class EdgeAwareGraphTransformer(nn.Module):
    """Edge-Aware Graph Transformer (EAGT)."""
    def __init__(self, node_dim, edge_dim, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_q = nn.Linear(node_dim, hidden_dim)
        self.W_k = nn.Linear(node_dim, hidden_dim)
        self.W_v = nn.Linear(node_dim, hidden_dim)
        self.W_e = nn.Linear(edge_dim, num_heads)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        Q = self.W_q(x).view(-1, self.num_heads, self.head_dim)
        K = self.W_k(x).view(-1, self.num_heads, self.head_dim)
        V = self.W_v(x).view(-1, self.num_heads, self.head_dim)
        
        src, dst = edge_index
        attn_scores = (Q[dst] * K[src]).sum(dim=-1) / self.scale + self.W_e(edge_attr)
        attn_expand = F.softmax(attn_scores, dim=0).unsqueeze(-1)
        weighted_v = attn_expand * V[src]
        
        out = torch.zeros(x.size(0), self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.head_dim), weighted_v)
        
        return self.norm(self.dropout(self.W_o(out.view(-1, self.num_heads * self.head_dim))))


class BidirectionalCrossModalAttention(nn.Module):
    """Bidirectional Cross-Modal Attention (BCMA)."""
    def __init__(self, dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.seq_to_struct = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.struct_to_seq = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.fusion_gate = nn.Sequential(nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.fusion_transform = nn.Sequential(nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq_feat, struct_feat):
        seq_feat, struct_feat = seq_feat.unsqueeze(1), struct_feat.unsqueeze(1)
        seq_enhanced, _ = self.seq_to_struct(seq_feat, struct_feat, struct_feat)
        struct_enhanced, _ = self.struct_to_seq(struct_feat, seq_feat, seq_feat)
        seq_enhanced, struct_enhanced = seq_enhanced.squeeze(1), struct_enhanced.squeeze(1)
        seq_feat, struct_feat = seq_feat.squeeze(1), struct_feat.squeeze(1)
        combined = torch.cat([seq_feat, struct_feat, seq_enhanced, struct_enhanced], dim=-1)
        gate = self.fusion_gate(combined)
        transform = self.fusion_transform(combined)
        return self.norm(gate * transform + (1 - gate) * (seq_feat + struct_feat) / 2)


class HierarchicalAttentionPooling(nn.Module):
    """Hierarchical Attention Pooling (HAP)."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.proj_max = nn.Linear(in_dim, hidden_dim)
        self.proj_mean = nn.Linear(in_dim, hidden_dim)
        self.proj_attn = nn.Linear(in_dim, hidden_dim)
        self.combine = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, batch):
        attn_scores = self.attention(x)
        attn_weights = torch.zeros_like(attn_scores)
        for i in torch.unique(batch):
            mask = batch == i
            attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)
        pool_attn = self.proj_attn(global_add_pool(x * attn_weights, batch))
        pool_max = self.proj_max(global_max_pool(x, batch))
        pool_mean = self.proj_mean(global_mean_pool(x, batch))
        return self.norm(self.combine(torch.cat([pool_max, pool_mean, pool_attn], dim=-1)))


class EnhancedSequenceEncoder(nn.Module):
    """Enhanced Sequence Encoder with BiLSTM + Self-Attention."""
    def __init__(self, vocab_size=66, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(self.embedding(x))
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        return self.norm(self.output_proj((lstm_out + attn_out).mean(dim=1)))


class OneHotCNNEncoder(nn.Module):
    """Multi-scale CNN Encoder for One-Hot encoding."""
    def __init__(self, input_dim=Config.NUM_CHARS, hidden_dim=128, output_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(hidden_dim), nn.BatchNorm1d(hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.combine = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h1, h2, h3 = F.gelu(self.bn1(self.conv1(x))).max(dim=-1)[0], F.gelu(self.bn2(self.conv2(x))).max(dim=-1)[0], F.gelu(self.bn3(self.conv3(x))).max(dim=-1)[0]
        return self.norm(self.combine(torch.cat([h1, h2, h3], dim=-1)))


# ============================================================
# Complete MMF-GNN Model
# ============================================================
class MMF_GNN(nn.Module):
    """MMF-GNN: Multi-Modal Fusion Graph Neural Network."""
    def __init__(self, node_dim=42, edge_dim=10, hidden_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.msga = MultiScaleGraphAttention(hidden_dim, hidden_dim // num_heads, num_heads, dropout)
        self.eagt = EdgeAwareGraphTransformer(hidden_dim, edge_dim, hidden_dim, num_heads, dropout)
        self.graph_combine = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.pool = HierarchicalAttentionPooling(hidden_dim, hidden_dim)
        self.seq_encoder = EnhancedSequenceEncoder(vocab_size=66, embed_dim=64, hidden_dim=hidden_dim, dropout=dropout)
        self.onehot_encoder = OneHotCNNEncoder(Config.NUM_CHARS, hidden_dim, hidden_dim, dropout)
        self.seq_combine = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.cross_modal = BidirectionalCrossModalAttention(hidden_dim, num_heads, dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(hidden_dim // 2, 1))

    def forward(self, x, edge_index, edge_attr, batch, label_enc, onehot_enc):
        h = self.node_proj(x)
        h_graph = self.graph_combine(torch.cat([self.msga(h, edge_index), self.eagt(h, edge_index, edge_attr, batch)], dim=-1))
        struct_feat = self.pool(h_graph, batch)
        seq_feat = self.seq_combine(torch.cat([self.seq_encoder(label_enc), self.onehot_encoder(onehot_enc)], dim=-1))
        return self.classifier(self.cross_modal(seq_feat, struct_feat))


# ============================================================
# Data Processing
# ============================================================
def prepare_data(df, verbose=True):
    """Prepare data from dataframe."""
    if verbose:
        print(f"\nProcessing {len(df)} molecules...")

    all_data, failed = [], 0
    for idx, row in df.iterrows():
        smiles, label = str(row['SMILES']), int(row['Activity'])
        x, edge_index, edge_attr = smiles_to_enhanced_graph(smiles)
        if x is None:
            failed += 1
            continue
        all_data.append({
            'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
            'label_enc': torch.tensor(smiles_to_label_encoding(smiles), dtype=torch.long),
            'onehot_enc': torch.tensor(smiles_to_onehot_encoding(smiles), dtype=torch.float),
            'label': label, 'smiles': smiles
        })
        if verbose and (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1} molecules...")

    labels = np.array([d['label'] for d in all_data])
    if verbose:
        print(f"\n✓ Processed {len(all_data)} molecules (Failed: {failed})")
        print(f"  Active: {sum(labels)}, Inactive: {len(labels) - sum(labels)}")
    return all_data, labels


def collate_fn(batch):
    """Custom collate function for batching."""
    x_list, edge_index_list, edge_attr_list = [d['x'] for d in batch], [d['edge_index'] for d in batch], [d['edge_attr'] for d in batch]
    batch_idx, offset, new_edge_indices, new_edge_attrs = [], 0, [], []
    
    for i, (x, edge_index, edge_attr) in enumerate(zip(x_list, edge_index_list, edge_attr_list)):
        batch_idx.extend([i] * x.size(0))
        new_edge_indices.append(edge_index + offset)
        new_edge_attrs.append(edge_attr)
        offset += x.size(0)
    
    return (torch.cat(x_list), torch.cat(new_edge_indices, dim=1), torch.cat(new_edge_attrs),
            torch.tensor(batch_idx, dtype=torch.long), torch.stack([d['label_enc'] for d in batch]),
            torch.stack([d['onehot_enc'] for d in batch]), torch.tensor([d['label'] for d in batch], dtype=torch.float))


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for batch in train_loader:
        x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        loss = criterion(model(x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc).view(-1), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, data_loader, device, threshold=0.5):
    """Evaluate model and compute metrics."""
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in data_loader:
            x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc, labels = [b.to(device) if i < 6 else b for i, b in enumerate(batch)]
            probs = torch.sigmoid(model(x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc).view(-1))
            y_true.extend(labels.numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true, y_prob = np.array(y_true), np.array(y_prob)
    y_pred = (y_prob > threshold).astype(int)
    
    return {
        'ACC': accuracy_score(y_true, y_pred), 'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUPRC': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
        'CK': cohen_kappa_score(y_true, y_pred), 'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
        'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred
    }


# ============================================================
# Data Loading Functions
# ============================================================
def load_and_merge_datasets(filepaths):
    """Load multiple datasets, merge them, and remove duplicate SMILES."""
    print("=" * 70)
    print("LOADING AND MERGING DATASETS")
    print("=" * 70)
    
    all_dfs = []
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"✗ File not found: {filepath}")
            continue
        
        df = pd.read_csv(filepath)
        
        # Handle different column names
        if 'canonical_smiles' in df.columns:
            df = df.rename(columns={'canonical_smiles': 'SMILES'})
        if 'Activity Class' in df.columns:
            df = df.rename(columns={'Activity Class': 'Activity'})
        if 'label' in df.columns and 'Activity' not in df.columns:
            df = df.rename(columns={'label': 'Activity'})
        
        if 'SMILES' in df.columns and 'Activity' in df.columns:
            df_clean = df[['SMILES', 'Activity']].dropna()
            df_clean['Activity'] = df_clean['Activity'].astype(int)
            df_clean['Source'] = os.path.basename(filepath)
            all_dfs.append(df_clean)
            print(f"✓ Loaded {os.path.basename(filepath)}: {len(df_clean)} samples")
        else:
            print(f"✗ Skipping {os.path.basename(filepath)}: Missing required columns")

    if not all_dfs:
        raise ValueError("No valid datasets loaded!")
    
    df_merged = pd.concat(all_dfs, ignore_index=True)
    total_before = len(df_merged)
    
    print(f"\n{'='*70}")
    print("DATA CLEANING")
    print(f"{'='*70}")
    print(f"Total samples before cleaning: {total_before}")
    
    # ===== CLEANING STEP 1: Remove empty/invalid SMILES =====
    df_merged = df_merged[df_merged['SMILES'].str.strip() != '']
    df_merged = df_merged[df_merged['SMILES'].notna()]
    print(f"After removing empty SMILES: {len(df_merged)}")
    
    # ===== CLEANING STEP 2: Standardize SMILES (strip whitespace) =====
    df_merged['SMILES'] = df_merged['SMILES'].str.strip()
    
    # ===== CLEANING STEP 3: Remove duplicate SMILES =====
    # Keep first occurrence (you can change to keep='last' if needed)
    duplicates_count = df_merged.duplicated(subset=['SMILES'], keep='first').sum()
    df_merged = df_merged.drop_duplicates(subset=['SMILES'], keep='first')
    print(f"Duplicate SMILES removed: {duplicates_count}")
    print(f"After removing duplicates: {len(df_merged)}")
    
    # ===== CLEANING STEP 4: Validate SMILES with RDKit =====
    print("Validating SMILES with RDKit...")
    valid_mask = []
    invalid_count = 0
    for smiles in df_merged['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mask.append(True)
        else:
            valid_mask.append(False)
            invalid_count += 1
    
    df_merged = df_merged[valid_mask]
    print(f"Invalid SMILES removed: {invalid_count}")
    print(f"After RDKit validation: {len(df_merged)}")
    
    # ===== CLEANING STEP 5: Check for conflicting labels =====
    # (Same SMILES with different Activity labels - already handled by drop_duplicates)
    
    # ===== Summary =====
    total_after = len(df_merged)
    total_removed = total_before - total_after
    
    print(f"\n{'='*70}")
    print("CLEANING SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples before: {total_before}")
    print(f"Total samples after:  {total_after}")
    print(f"Total removed:        {total_removed} ({100*total_removed/total_before:.2f}%)")
    print(f"\nFinal dataset:")
    print(f"  Active:   {df_merged['Activity'].sum()}")
    print(f"  Inactive: {len(df_merged) - df_merged['Activity'].sum()}")
    print(f"  Active ratio: {df_merged['Activity'].mean()*100:.2f}%")
    
    # Show samples per source file
    print(f"\nSamples per source (after cleaning):")
    for source in df_merged['Source'].unique():
        count = len(df_merged[df_merged['Source'] == source])
        print(f"  {source}: {count}")
    
    # Drop the Source column before returning
    df_merged = df_merged.drop(columns=['Source'])
    
    return df_merged


# ============================================================
# Plotting Functions
# ============================================================
def generate_plots(final_metrics, history, output_dir):
    """Generate and save all plots separately."""
    print("\nGenerating separate plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
    
    # Ensure epochs range list applies consistently
    epochs_eval = history.get('eval_epochs', range(10, len(history['test_acc']) * 10 + 1, 10))

    # Plot 1: Training Loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300)
    plt.close()

    # Plot 2: Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_eval, history['test_acc'], 'g-', label='Accuracy', linewidth=2, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Score')
    plt.title('Test Accuracy Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_over_epochs.png'), dpi=300)
    plt.close()

    # Plot 3: F1 Score
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_eval, history['test_f1'], 'r-', label='F1 Score', linewidth=2, marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Test F1 Score Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_over_epochs.png'), dpi=300)
    plt.close()

    # Plot 4: AUPRC over Epochs
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_eval, history['test_auprc'], 'purple', label='AUPRC', linewidth=2, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.title('Test AUPRC Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auprc_over_epochs.png'), dpi=300)
    plt.close()

    # Plot 5: MCC over Epochs
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_eval, history['test_mcc'], 'orange', label='MCC', linewidth=2, marker='d')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Test MCC Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mcc_over_epochs.png'), dpi=300)
    plt.close()

    # Plot 6: Final ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(final_metrics['y_true'], final_metrics['y_prob'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Final ROC Curve', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    # Plot 7: Final Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(final_metrics['y_true'], final_metrics['y_prob'])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, 'g-', linewidth=2, label=f'PR (AUPRC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Final Precision-Recall Curve', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve_final.png'), dpi=300)
    plt.close()

    # Plot 8: Precision-Recall Curves over Epochs (Overlayed)
    if 'y_true_epochs' in history and 'y_prob_epochs' in history:
        plt.figure(figsize=(9, 7))
        cm = plt.get_cmap('viridis')
        num_evals = len(history['eval_epochs'])
        
        for i, (ep, y_t, y_p) in enumerate(zip(history['eval_epochs'], history['y_true_epochs'], history['y_prob_epochs'])):
            prec, rec, _ = precision_recall_curve(y_t, y_p)
            pr_a = auc(rec, prec)
            color = cm(i / max(1, num_evals - 1))
            plt.plot(rec, prec, color=color, alpha=0.8, linewidth=1.5, label=f'Epoch {ep:03d} (AUPRC={pr_a:.3f})')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Over Epochs', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pr_curves_over_epochs.png'), dpi=300)
        plt.close()

    # Plot 9: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_mat = confusion_matrix(final_metrics['y_true'], final_metrics['y_pred'])
    sns.heatmap(cm_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Final Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Plot 10: Distribution of Predicted Probabilities
    plt.figure(figsize=(8, 6))
    y_true_arr = np.array(final_metrics['y_true'])
    y_prob_arr = np.array(final_metrics['y_prob'])
    
    sns.histplot(y_prob_arr[y_true_arr == 0], color='blue', label='Inactive (True Label)', 
                 kde=True, stat='density', bins=30, alpha=0.5)
    sns.histplot(y_prob_arr[y_true_arr == 1], color='red', label='Active (True Label)', 
                 kde=True, stat='density', bins=30, alpha=0.5)
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities by True Class', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prob_distribution.png'), dpi=300)
    plt.close()

    print(f"✓ All separated plots saved to {output_dir}/")


# ============================================================
# Main Training Function
# ============================================================
def train(args):
    """Main training function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("MMF-GNN: Multi-Modal Fusion Graph Neural Network")
    print(f"{'='*70}")
    print(f"Device: {device} | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load data
    df_merged = load_and_merge_datasets(args.data_path)
    all_data, labels = prepare_data(df_merged)

    # 80-20 Split
    print(f"\n{'='*70}\nDATA SPLIT (80-20)\n{'='*70}")
    train_idx, test_idx = train_test_split(np.arange(len(all_data)), test_size=args.test_size, stratify=labels, random_state=args.seed)
    train_data, test_data = [all_data[i] for i in train_idx], [all_data[i] for i in test_idx]
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # Balance training data
    train_inactive = [d for d in train_data if d['label'] == 0]
    train_active = [d for d in train_data if d['label'] == 1]
    print(f"Training - Active: {len(train_active)}, Inactive: {len(train_inactive)}")

    if len(train_inactive) > 0 and len(train_active) > len(train_inactive):
        train_balanced = train_active + train_inactive * max(1, len(train_active) // len(train_inactive))
    elif len(train_active) > 0 and len(train_inactive) > len(train_active):
        train_balanced = train_inactive + train_active * max(1, len(train_inactive) // len(train_active))
    else:
        train_balanced = train_data
    np.random.shuffle(train_balanced)
    print(f"After balancing: {len(train_balanced)}")

    # Data loaders
    train_loader = DataLoader(train_balanced, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = MMF_GNN(node_dim=Config.NODE_FEATURES, edge_dim=Config.EDGE_FEATURES, 
                    hidden_dim=args.hidden_dim, num_heads=args.num_heads, dropout=args.dropout).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"\n{'='*70}\nTRAINING\n{'='*70}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    best_auprc, best_state = 0, None
    history = {
        'train_loss': [], 'test_acc': [], 'test_f1': [], 'test_auprc': [], 'test_mcc': [],
        'eval_epochs': [], 'y_true_epochs': [], 'y_prob_epochs': []
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        history['train_loss'].append(train_loss)

        if epoch % 10 == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, device)
            
            history['eval_epochs'].append(epoch)
            history['test_acc'].append(metrics['ACC'])
            history['test_f1'].append(metrics['F1'])
            history['test_auprc'].append(metrics['AUPRC'])
            history['test_mcc'].append(metrics['MCC'])
            history['y_true_epochs'].append(metrics['y_true'])
            history['y_prob_epochs'].append(metrics['y_prob'])
            
            print(f"Epoch {epoch:3d}/{args.epochs}: Loss={train_loss:.4f} | ACC={metrics['ACC']:.4f}, F1={metrics['F1']:.4f}, AUPRC={metrics['AUPRC']:.4f}, MCC={metrics['MCC']:.4f}")
            
            if metrics['AUPRC'] > best_auprc:
                best_auprc = metrics['AUPRC']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"         ↑ New best model! (AUPRC: {best_auprc:.4f})")

    # Load best model
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final evaluation
    print(f"\n{'='*70}\nFINAL EVALUATION\n{'='*70}")
    final_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTEST RESULTS:")
    print(f"  ACC: {final_metrics['ACC']:.4f} | F1: {final_metrics['F1']:.4f} | AUPRC: {final_metrics['AUPRC']:.4f}")
    print(f"  AUC: {final_metrics['AUC']:.4f} | CK: {final_metrics['CK']:.4f} | MCC: {final_metrics['MCC']:.4f}")
    
    print(f"\n{classification_report(final_metrics['y_true'], final_metrics['y_pred'], target_names=['Inactive', 'Active'], zero_division=0)}")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Do not save full lists of probabilities in the primary PT history object to avoid inflation in large datasets
    history_save = {k: v for k, v in history.items() if k not in ['y_true_epochs', 'y_prob_epochs']}
    
    torch.save({'model_state_dict': model.state_dict(), 'final_metrics': final_metrics, 
                'history': history_save, 'config': vars(args)}, os.path.join(args.output_dir, 'mmf_gnn_model.pt'))
    
    pd.DataFrame({'Metric': ['ACC', 'F1', 'AUPRC', 'AUC', 'CK', 'MCC'],
                  'Value': [final_metrics['ACC'], final_metrics['F1'], final_metrics['AUPRC'],
                           final_metrics['AUC'], final_metrics['CK'], final_metrics['MCC']]}).to_csv(os.path.join(args.output_dir, 'train4results.csv'), index=False)
    
    pd.DataFrame({'True_Label': final_metrics['y_true'], 'Predicted_Label': final_metrics['y_pred'],
                  'Probability': final_metrics['y_prob']}).to_csv(os.path.join(args.output_dir, 'train4predictions.csv'), index=False)
    
    # Generate all requested separate plots
    generate_plots(final_metrics, history, args.output_dir)
    
    print(f"\n{'='*70}\nTRAINING COMPLETE!\n{'='*70}")


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMF-GNN Training')
    parser.add_argument('--data_path', type=str, nargs='+', required=True, help='Path(s) to data CSV file(s)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (0, 1, or 2)')
    
    args = parser.parse_args()
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"\n✓ Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    
    train(args)