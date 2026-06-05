#!/usr/bin/env python3
"""
MMF-GNN: Inference/Test Script

Load a trained MMF-GNN model and make predictions on new data.

Usage:
    python test.py --model_path ./output/mmf_gnn_model.pt --data_path /path/to/test_data.csv --output_dir ./predictions

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
from sklearn.metrics import (accuracy_score, f1_score, average_precision_score,
                             cohen_kappa_score, matthews_corrcoef,
                             classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve, auc)
import matplotlib
matplotlib.use('Agg')
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
# Model Components (Same as train.py)
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
def prepare_data(df, has_labels=True, verbose=True):
    """Prepare data from dataframe."""
    if verbose:
        print(f"\nProcessing {len(df)} molecules...")

    all_data, failed, failed_smiles = [], 0, []
    for idx, row in df.iterrows():
        smiles = str(row['SMILES'])
        label = int(row['Activity']) if has_labels and 'Activity' in df.columns else -1
        
        x, edge_index, edge_attr = smiles_to_enhanced_graph(smiles)
        if x is None:
            failed += 1
            failed_smiles.append(smiles)
            continue
            
        all_data.append({
            'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
            'label_enc': torch.tensor(smiles_to_label_encoding(smiles), dtype=torch.long),
            'onehot_enc': torch.tensor(smiles_to_onehot_encoding(smiles), dtype=torch.float),
            'label': label, 'smiles': smiles
        })
        
        if verbose and (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1} molecules...")

    if verbose:
        print(f"\n✓ Processed {len(all_data)} molecules (Failed: {failed})")
        if has_labels:
            labels = [d['label'] for d in all_data]
            print(f"  Active: {sum(labels)}, Inactive: {len(labels) - sum(labels)}")
    
    return all_data, failed_smiles


def collate_fn(batch):
    """Custom collate function for batching."""
    x_list = [d['x'] for d in batch]
    edge_index_list = [d['edge_index'] for d in batch]
    edge_attr_list = [d['edge_attr'] for d in batch]
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
# Load Data
# ============================================================
def load_test_data(filepath):
    """Load test data from CSV."""
    print(f"\n{'='*70}")
    print("LOADING TEST DATA")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Handle different column names
    if 'canonical_smiles' in df.columns:
        df = df.rename(columns={'canonical_smiles': 'SMILES'})
    if 'Activity Class' in df.columns:
        df = df.rename(columns={'Activity Class': 'Activity'})
    if 'label' in df.columns and 'Activity' not in df.columns:
        df = df.rename(columns={'label': 'Activity'})
    
    if 'SMILES' not in df.columns:
        raise ValueError("CSV must have a 'SMILES' column!")
    
    has_labels = 'Activity' in df.columns
    
    print(f"✓ Loaded {len(df)} samples from {os.path.basename(filepath)}")
    print(f"  Has labels: {has_labels}")
    
    if has_labels:
        df = df.dropna(subset=['SMILES', 'Activity'])
        df['Activity'] = df['Activity'].astype(int)
        print(f"  Active: {df['Activity'].sum()}, Inactive: {len(df) - df['Activity'].sum()}")
    else:
        df = df.dropna(subset=['SMILES'])
    
    return df, has_labels


# ============================================================
# Inference Function
# ============================================================
def predict(model, data_loader, device):
    """Run inference and return predictions."""
    model.eval()
    all_smiles, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in data_loader:
            x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc, labels = batch
            
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            batch_idx = batch_idx.to(device)
            label_enc = label_enc.to(device)
            onehot_enc = onehot_enc.to(device)
            
            logits = model(x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc).view(-1)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_probs), np.array(all_labels)


# ============================================================
# Generate Plots (for labeled data)
# ============================================================
def generate_test_plots(y_true, y_prob, y_pred, output_dir):
    """Generate evaluation plots for test data."""
    print("\nGenerating plots...")
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_roc_curve.png'), dpi=300)
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, 'g-', linewidth=2, label=f'PR (AUPRC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_pr_curve.png'), dpi=300)
    plt.close()
    
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Probability Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(y_prob[y_true == 0], color='blue', label='Inactive', kde=True, stat='density', bins=30, alpha=0.5)
    sns.histplot(y_prob[y_true == 1], color='red', label='Active', kde=True, stat='density', bins=30, alpha=0.5)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution by Class', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_prob_distribution.png'), dpi=300)
    plt.close()
    
    print(f"✓ Plots saved to {output_dir}/")


# ============================================================
# Main Test Function
# ============================================================
def test(args):
    """Main test/inference function."""
    
    print(f"\n{'='*70}")
    print("MMF-GNN: INFERENCE MODE")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available() and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        print(f"✓ Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print(f"✓ Using device: {device}")
    
    # Load model checkpoint
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    print(f"✓ Loaded model from: {args.model_path}")
    print(f"  Hidden dim: {config.get('hidden_dim', 128)}")
    print(f"  Num heads: {config.get('num_heads', 4)}")
    print(f"  Dropout: {config.get('dropout', 0.3)}")
    
    # Initialize model
    model = MMF_GNN(
        node_dim=Config.NODE_FEATURES,
        edge_dim=Config.EDGE_FEATURES,
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load test data
    df, has_labels = load_test_data(args.data_path)
    all_data, failed_smiles = prepare_data(df, has_labels=has_labels)
    
    # Create DataLoader
    test_loader = DataLoader(all_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Run inference
    print(f"\n{'='*70}")
    print("RUNNING INFERENCE")
    print(f"{'='*70}")
    
    probs, labels = predict(model, test_loader, device)
    preds = (probs > args.threshold).astype(int)
    
    # Get SMILES for output
    smiles_list = [d['smiles'] for d in all_data]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save predictions
    results_df = pd.DataFrame({
        'SMILES': smiles_list,
        'Probability': probs,
        'Predicted_Label': preds,
        'Predicted_Class': ['Active' if p == 1 else 'Inactive' for p in preds]
    })
    
    if has_labels:
        results_df['True_Label'] = labels
        results_df['True_Class'] = ['Active' if l == 1 else 'Inactive' for l in labels]
        results_df['Correct'] = (preds == labels)
    
    results_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    print(f"\n✓ Predictions saved to {args.output_dir}/predictions.csv")
    
    # Save failed SMILES
    if failed_smiles:
        pd.DataFrame({'Failed_SMILES': failed_smiles}).to_csv(
            os.path.join(args.output_dir, 'failed_smiles.csv'), index=False)
        print(f"✓ Failed SMILES saved to {args.output_dir}/failed_smiles.csv")
    
    # Print summary
    print(f"\n{'='*70}")
    print("PREDICTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total molecules: {len(smiles_list)}")
    print(f"Predicted Active: {sum(preds)} ({100*sum(preds)/len(preds):.2f}%)")
    print(f"Predicted Inactive: {len(preds) - sum(preds)} ({100*(len(preds)-sum(preds))/len(preds):.2f}%)")
    print(f"Threshold: {args.threshold}")
    
    # If labels available, compute metrics
    if has_labels and len(np.unique(labels)) > 1:
        print(f"\n{'='*70}")
        print("EVALUATION METRICS")
        print(f"{'='*70}")
        
        metrics = {
            'ACC': accuracy_score(labels, preds),
            'F1': f1_score(labels, preds, zero_division=0),
            'AUPRC': average_precision_score(labels, probs),
            'AUC': roc_auc_score(labels, probs),
            'CK': cohen_kappa_score(labels, preds),
            'MCC': matthews_corrcoef(labels, preds)
        }
        
        print(f"\n  ACC:   {metrics['ACC']:.4f}")
        print(f"  F1:    {metrics['F1']:.4f}")
        print(f"  AUPRC: {metrics['AUPRC']:.4f}")
        print(f"  AUC:   {metrics['AUC']:.4f}")
        print(f"  CK:    {metrics['CK']:.4f}")
        print(f"  MCC:   {metrics['MCC']:.4f}")
        
        print(f"\n{classification_report(labels, preds, target_names=['Inactive', 'Active'], zero_division=0)}")
        
        # Save metrics
        pd.DataFrame({'Metric': list(metrics.keys()), 'Value': list(metrics.values())}).to_csv(
            os.path.join(args.output_dir, 'test_metrics.csv'), index=False)
        
        # Generate plots
        generate_test_plots(labels, probs, preds, args.output_dir)
    
    print(f"\n{'='*70}")
    print("INFERENCE COMPLETE!")
    print(f"{'='*70}")


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMF-GNN Inference/Test')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    test(args)