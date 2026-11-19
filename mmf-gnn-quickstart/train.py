#!/usr/bin/env python3
"""
MMF-GNN: Multi-Modal Fusion Graph Neural Network Training Script

Usage:
    python train.py --data_path sample_dataset.csv --output_dir ./outputs --epochs 100
    python train.py --data_path data1.csv data2.csv --output_dir ./outputs --epochs 100
"""

import os
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from mmf_gnn import Config, MMF_GNN, prepare_data, collate_fn, evaluate


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for batch in train_loader:
        x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc, labels = [
            b.to(device) for b in batch
        ]
        optimizer.zero_grad()
        loss = criterion(model(x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc).view(-1), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


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

    df_merged = df_merged[df_merged['SMILES'].str.strip() != '']
    df_merged = df_merged[df_merged['SMILES'].notna()]
    print(f"After removing empty SMILES: {len(df_merged)}")

    df_merged['SMILES'] = df_merged['SMILES'].str.strip()

    duplicates_count = df_merged.duplicated(subset=['SMILES'], keep='first').sum()
    df_merged = df_merged.drop_duplicates(subset=['SMILES'], keep='first')
    print(f"Duplicate SMILES removed: {duplicates_count}")
    print(f"After removing duplicates: {len(df_merged)}")

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

    print(f"\nSamples per source (after cleaning):")
    for source in df_merged['Source'].unique():
        count = len(df_merged[df_merged['Source'] == source])
        print(f"  {source}: {count}")

    df_merged = df_merged.drop(columns=['Source'])
    return df_merged


# ============================================================
# Output Directory Validation Helper
# ============================================================
def ensure_output_dir(output_dir):
    """Ensure output directory exists and is writable, falling back if permission is denied."""
    folder_name = os.path.basename(os.path.normpath(output_dir))
    if not folder_name or folder_name in ('.', '..'):
        folder_name = 'output'

    candidates = [
        output_dir,
        os.path.join(os.getcwd(), folder_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name),
        os.path.join('/tmp', folder_name),
    ]

    for idx, path in enumerate(candidates):
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('')
            os.remove(test_file)
            if idx > 0:
                print(f"⚠ Warning: Permission denied for '{output_dir}'. Using writable directory: '{path}'")
            return path
        except (PermissionError, OSError):
            continue

    raise PermissionError(f"Unable to create or write to output directory '{output_dir}' or any fallback directory.")


# ============================================================
# Plotting Functions
# ============================================================
def generate_plots(final_metrics, history, output_dir):
    """Generate and save all plots separately."""
    print("\nGenerating separate plots...")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')

    epochs_eval = history.get('eval_epochs', range(10, len(history['test_acc']) * 10 + 1, 10))
    legend_font = {'size': 16, 'weight': 'bold'}

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=22, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=22, fontweight='bold')
    plt.title('Training Loss Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'training_loss.pdf'), dpi=600)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_eval, history['test_acc'], 'g-', label='Accuracy', linewidth=2, marker='s')
    plt.xlabel('Epoch', fontsize=22, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=22, fontweight='bold')
    plt.title('Test Accuracy Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(prop=legend_font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_over_epochs.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'accuracy_over_epochs.pdf'), dpi=600)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_eval, history['test_f1'], 'r-', label='F1 Score', linewidth=2, marker='^')
    plt.xlabel('Epoch', fontsize=22, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=22, fontweight='bold')
    plt.title('Test F1 Score Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(prop=legend_font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_over_epochs.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'f1_score_over_epochs.pdf'), dpi=600)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_eval, history['test_auprc'], 'purple', label='AUPRC', linewidth=2, marker='o')
    plt.xlabel('Epoch', fontsize=22, fontweight='bold')
    plt.ylabel('AUPRC', fontsize=22, fontweight='bold')
    plt.title('Test AUPRC Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(prop=legend_font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auprc_over_epochs.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'auprc_over_epochs.pdf'), dpi=600)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_eval, history['test_mcc'], 'orange', label='MCC', linewidth=2, marker='d')
    plt.xlabel('Epoch', fontsize=22, fontweight='bold')
    plt.ylabel('MCC', fontsize=22, fontweight='bold')
    plt.title('Test MCC Over Epochs', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(prop=legend_font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mcc_over_epochs.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'mcc_over_epochs.pdf'), dpi=600)
    plt.close()

    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(final_metrics['y_true'], final_metrics['y_prob'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate', fontsize=22, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=22, fontweight='bold')
    plt.title('Final ROC Curve', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', prop=legend_font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'roc_curve.pdf'), dpi=600)
    plt.close()

    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(final_metrics['y_true'], final_metrics['y_prob'])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, 'g-', linewidth=2, label=f'PR (AUPRC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=22, fontweight='bold')
    plt.ylabel('Precision', fontsize=22, fontweight='bold')
    plt.title('Final Precision-Recall Curve', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', prop=legend_font)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve_final.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'pr_curve_final.pdf'), dpi=600)
    plt.close()

    if 'y_true_epochs' in history and 'y_prob_epochs' in history:
        plt.figure(figsize=(9, 7))
        cm = plt.get_cmap('viridis')
        num_evals = len(history['eval_epochs'])

        for i, (ep, y_t, y_p) in enumerate(
            zip(history['eval_epochs'], history['y_true_epochs'], history['y_prob_epochs'])
        ):
            prec, rec, _ = precision_recall_curve(y_t, y_p)
            pr_a = auc(rec, prec)
            color = cm(i / max(1, num_evals - 1))
            plt.plot(rec, prec, color=color, alpha=0.8, linewidth=1.5, label=f'Epoch {ep:03d} (AUPRC={pr_a:.3f})')

        plt.xlabel('Recall', fontsize=22, fontweight='bold')
        plt.ylabel('Precision', fontsize=22, fontweight='bold')
        plt.title('Precision-Recall Curves Over Epochs', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=legend_font)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pr_curves_over_epochs.png'), dpi=600)
        plt.savefig(os.path.join(output_dir, 'pr_curves_over_epochs.pdf'), dpi=600)
        plt.close()

    plt.figure(figsize=(6, 5))
    cm_mat = confusion_matrix(final_metrics['y_true'], final_metrics['y_pred'])
    sns.heatmap(
        cm_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Inactive', 'Active'],
        yticklabels=['Inactive', 'Active'],
    )
    plt.xlabel('Predicted Label', fontsize=22, fontweight='bold')
    plt.ylabel('True Label', fontsize=22, fontweight='bold')
    plt.title('Final Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'), dpi=600)
    plt.close()

    plt.figure(figsize=(8, 6))
    y_true_arr = np.array(final_metrics['y_true'])
    y_prob_arr = np.array(final_metrics['y_prob'])

    sns.histplot(
        y_prob_arr[y_true_arr == 0],
        color='blue',
        label='Inactive (True Label)',
        kde=True,
        stat='density',
        bins=30,
        alpha=0.5,
    )
    sns.histplot(
        y_prob_arr[y_true_arr == 1],
        color='red',
        label='Active (True Label)',
        kde=True,
        stat='density',
        bins=30,
        alpha=0.5,
    )

    plt.xlabel('Predicted Probability', fontsize=22, fontweight='bold')
    plt.ylabel('Density', fontsize=22, fontweight='bold')
    plt.title('Distribution of Predicted Probabilities by True Class', fontweight='bold')
    plt.legend(prop=legend_font)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prob_distribution.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, 'prob_distribution.pdf'), dpi=600)
    plt.close()

    print(f"✓ All separated plots saved to {output_dir}/")


# ============================================================
# Main Training Function
# ============================================================
def train(args):
    """Main training function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output_dir = ensure_output_dir(args.output_dir)

    print(f"\n{'='*70}")
    print("MMF-GNN: Multi-Modal Fusion Graph Neural Network")
    print(f"{'='*70}")
    print(f"Device: {device} | Output Dir: {args.output_dir} | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    df_merged = load_and_merge_datasets(args.data_path)
    all_data, labels = prepare_data(df_merged)

    print(f"\n{'='*70}\nDATA SPLIT (80-20)\n{'='*70}")
    train_idx, test_idx = train_test_split(
        np.arange(len(all_data)), test_size=args.test_size, stratify=labels, random_state=args.seed
    )
    train_data, test_data = [all_data[i] for i in train_idx], [all_data[i] for i in test_idx]
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

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

    train_loader = DataLoader(
        train_balanced, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MMF_GNN(
        node_dim=Config.NODE_FEATURES,
        edge_dim=Config.EDGE_FEATURES,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n{'='*70}\nTRAINING\n{'='*70}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}, Early Stopping Patience: {args.patience}")

    best_auprc, best_state, best_epoch = 0, None, 0
    history = {
        'train_loss': [],
        'test_acc': [],
        'test_f1': [],
        'test_auprc': [],
        'test_mcc': [],
        'eval_epochs': [],
        'y_true_epochs': [],
        'y_prob_epochs': [],
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

            print(
                f"Epoch {epoch:3d}/{args.epochs}: Loss={train_loss:.4f} | "
                f"ACC={metrics['ACC']:.4f}, F1={metrics['F1']:.4f}, "
                f"AUPRC={metrics['AUPRC']:.4f}, MCC={metrics['MCC']:.4f}"
            )

            if metrics['AUPRC'] > best_auprc:
                best_auprc = metrics['AUPRC']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                print(f"         ↑ New best model! (AUPRC: {best_auprc:.4f})")
            elif args.patience > 0 and (epoch - best_epoch) >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch} (no AUPRC improvement for {epoch - best_epoch} epochs).")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print(f"\n{'='*70}\nFINAL EVALUATION\n{'='*70}")
    final_metrics = evaluate(model, test_loader, device)

    print(f"\nTEST RESULTS:")
    print(f"  ACC: {final_metrics['ACC']:.4f} | F1: {final_metrics['F1']:.4f} | AUPRC: {final_metrics['AUPRC']:.4f}")
    print(f"  AUC: {final_metrics['AUC']:.4f} | CK: {final_metrics['CK']:.4f} | MCC: {final_metrics['MCC']:.4f}")

    print(f"\n{classification_report(final_metrics['y_true'], final_metrics['y_pred'], target_names=['Inactive', 'Active'], zero_division=0)}")

    os.makedirs(args.output_dir, exist_ok=True)
    history_save = {k: v for k, v in history.items() if k not in ['y_true_epochs', 'y_prob_epochs']}

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'final_metrics': final_metrics,
            'history': history_save,
            'config': vars(args),
        },
        os.path.join(args.output_dir, 'mmf_gnn_model.pt'),
    )

    pd.DataFrame(
        {
            'Metric': ['ACC', 'F1', 'AUPRC', 'AUC', 'CK', 'MCC'],
            'Value': [
                final_metrics['ACC'],
                final_metrics['F1'],
                final_metrics['AUPRC'],
                final_metrics['AUC'],
                final_metrics['CK'],
                final_metrics['MCC'],
            ],
        }
    ).to_csv(os.path.join(args.output_dir, 'train4results.csv'), index=False)

    pd.DataFrame(
        {
            'True_Label': final_metrics['y_true'],
            'Predicted_Label': final_metrics['y_pred'],
            'Probability': final_metrics['y_prob'],
        }
    ).to_csv(os.path.join(args.output_dir, 'train4predictions.csv'), index=False)

    generate_plots(final_metrics, history, args.output_dir)

    print(f"\n{'='*70}\nTRAINING COMPLETE!\n{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMF-GNN Training')
    parser.add_argument('--data_path', type=str, nargs='+', required=True, help='Path(s) to data CSV file(s)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience in epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"\n✓ Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    train(args)
