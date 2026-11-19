#!/usr/bin/env python3
"""
MMF-GNN Quick Start & Installation Verification Script

Verifies that all required dependencies (PyTorch, PyTorch Geometric, RDKit,
scikit-learn, pandas, matplotlib, seaborn) are properly installed and tests
the full MMF-GNN data featurization, training, checkpointing, and inference
pipeline on a small sample dataset without downloading the full dataset.

Usage:
    python quick_start.py
    python quick_start.py --epochs 5 --output_dir quickstart_output
    python quick_start.py --checkpoint path/to/mmf_gnn_model.pt --skip_train
"""

import os
import sys
import argparse
import importlib
from datetime import datetime


REQUIRED_PACKAGES = [
    ("torch", "PyTorch"),
    ("torch_geometric", "PyTorch Geometric"),
    ("rdkit", "RDKit"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
]


def verify_dependencies():
    """Step 1: Verify all required Python libraries are importable."""
    print("\n" + "=" * 70)
    print("STEP 1/5: VERIFYING ENVIRONMENT & DEPENDENCIES")
    print("=" * 70)

    missing = []
    for module_name, label in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  [PASS] {label:<20} ({module_name} == {version})")
        except ImportError as exc:
            print(f"  [FAIL] {label:<20} ({module_name}: {exc})")
            missing.append(module_name)

    if missing:
        print("\nMissing dependencies detected:", ", ".join(missing))
        print("Install them with: pip install -r requirements.txt")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = torch.cuda.get_device_name(0) if device == "cuda" else "CPU execution"
    print(f"  [INFO] Compute device: {device} ({gpu_info})")
    return device


def main():
    parser = argparse.ArgumentParser(
        description="Verify MMF-GNN installation and end-to-end inference pipeline."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--sample_data",
        type=str,
        default=os.path.join(script_dir, "sample_dataset.csv"),
        help="Path to the small sample CSV dataset (columns: SMILES, Activity).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(script_dir, "quickstart_output"),
        help="Directory where verification artifacts and predictions are saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of smoke-training epochs when --skip_train is not set.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for quick start.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for smoke test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional pre-trained .pt checkpoint to verify inference directly.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip smoke training and run inference directly using --checkpoint.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MMF-GNN QUICK START VERIFICATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Step 1: Dependencies
    device = verify_dependencies()

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    from mmf_gnn import (
        Config,
        MMF_GNN,
        prepare_data,
        collate_fn,
        evaluate,
        load_model_checkpoint,
        predict_smiles,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 2: Sample Dataset & Featurization
    print("\n" + "=" * 70)
    print("STEP 2/5: LOADING SAMPLE DATASET & GRAPH/SEQUENCE FEATURIZATION")
    print("=" * 70)

    if not os.path.exists(args.sample_data):
        raise FileNotFoundError(f"Sample dataset not found: {args.sample_data}")

    df = pd.read_csv(args.sample_data)
    if "SMILES" not in df.columns or "Activity" not in df.columns:
        raise ValueError("Sample dataset must contain 'SMILES' and 'Activity' columns.")

    all_data, labels = prepare_data(df, has_labels=True, verbose=True)
    if len(all_data) < 4:
        raise RuntimeError("Sample dataset yielded fewer than 4 valid molecules.")

    sample_item = all_data[0]
    assert sample_item["x"].shape[1] == Config.NODE_FEATURES
    assert sample_item["edge_attr"].shape[1] == Config.EDGE_FEATURES
    assert sample_item["label_enc"].shape[0] == Config.MAX_SMILES_LEN
    assert sample_item["onehot_enc"].shape == (Config.MAX_SMILES_LEN, Config.NUM_CHARS)
    print(
        f"  [PASS] Tensor shapes verified: "
        f"x={tuple(sample_item['x'].shape)}, "
        f"edge_attr={tuple(sample_item['edge_attr'].shape)}, "
        f"label_enc={tuple(sample_item['label_enc'].shape)}, "
        f"onehot_enc={tuple(sample_item['onehot_enc'].shape)}"
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(all_data)),
        test_size=0.25,
        stratify=labels,
        random_state=args.seed,
    )
    train_data = [all_data[i] for i in train_idx]
    test_data = [all_data[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Step 3 & 4: Model Instantiation + Smoke Training (or Checkpoint Load)
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, "quickstart_model.pt")

    if args.skip_train:
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise FileNotFoundError("--skip_train requires an existing --checkpoint file.")
        print("\n" + "=" * 70)
        print("STEP 3/5 & 4/5: LOADING PRE-TRAINED CHECKPOINT")
        print("=" * 70)
        model, _ = load_model_checkpoint(args.checkpoint, device=device)
        print(f"  [PASS] Loaded checkpoint from {args.checkpoint}")
    else:
        print("\n" + "=" * 70)
        print("STEP 3/5: INSTANTIATING MMF-GNN & VERIFYING FORWARD PASS")
        print("=" * 70)
        model = MMF_GNN(
            node_dim=Config.NODE_FEATURES,
            edge_dim=Config.EDGE_FEATURES,
            hidden_dim=128,
            num_heads=4,
            dropout=0.3,
        ).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  [PASS] MMF-GNN initialized ({param_count:,} trainable parameters)")

        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False
        )
        first_batch = next(iter(train_loader))
        bx, bei, bea, bidx, blab, boh, _ = [b.to(device) for b in first_batch]
        with torch.no_grad():
            model.eval()
            out = model(bx, bei, bea, bidx, blab, boh)
            assert out.shape == (blab.size(0), 1)
        print(f"  [PASS] Forward pass verified: batch_size={blab.size(0)} -> output={tuple(out.shape)}")

        print("\n" + "=" * 70)
        print(f"STEP 4/5: RUNNING {args.epochs}-EPOCH SMOKE TRAINING & SAVING CHECKPOINT")
        print("=" * 70)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc, y = [
                    b.to(device) for b in batch
                ]
                optimizer.zero_grad()
                logits = model(x, edge_index, edge_attr, batch_idx, label_enc, onehot_enc).view(-1)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss.item())
            avg_loss = epoch_loss / max(1, len(train_loader))
            print(f"  Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {"hidden_dim": 128, "num_heads": 4, "dropout": 0.3},
                "verified_at": datetime.now().isoformat(),
            },
            checkpoint_path,
        )
        print(f"  [PASS] Checkpoint saved to {checkpoint_path}")

    # Step 5: Reload Checkpoint & Verify End-to-End Inference Pipeline
    print("\n" + "=" * 70)
    print("STEP 5/5: VERIFYING CHECKPOINT RELOAD & INFERENCE PIPELINE")
    print("=" * 70)

    reloaded_model, _ = load_model_checkpoint(checkpoint_path, device=device)
    metrics = evaluate(reloaded_model, test_loader, device=device)
    print(
        f"  Holdout Sample Metrics ({len(test_data)} molecules): "
        f"ACC={metrics['ACC']:.4f} | F1={metrics['F1']:.4f} | "
        f"AUPRC={metrics['AUPRC']:.4f} | AUC={metrics['AUC']:.4f} | MCC={metrics['MCC']:.4f}"
    )

    test_smiles = [d["smiles"] for d in test_data]
    preds = predict_smiles(reloaded_model, test_smiles, device=device)
    for row, true_item in zip(preds, test_data):
        row["True_Label"] = int(true_item["label"])

    preds_df = pd.DataFrame(preds)
    preds_path = os.path.join(args.output_dir, "quickstart_predictions.csv")
    preds_df.to_csv(preds_path, index=False)

    metrics_df = pd.DataFrame(
        {
            "Metric": ["ACC", "F1", "AUPRC", "AUC", "CK", "MCC"],
            "Value": [
                metrics["ACC"],
                metrics["F1"],
                metrics["AUPRC"],
                metrics["AUC"],
                metrics["CK"],
                metrics["MCC"],
            ],
        }
    )
    metrics_path = os.path.join(args.output_dir, "quickstart_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n  Sample Inference Output (first 5 holdout molecules):")
    for row in preds[:5]:
        print(
            f"    True={row['True_Label']} | Pred={row['Predicted_Label']} ({row['Predicted_Class']:<8}) "
            f"| P(Active)={row['Probability']:.4f} | SMILES={row['SMILES'][:45]}"
        )

    print(f"\n  [PASS] Saved predictions: {preds_path}")
    print(f"  [PASS] Saved metrics:     {metrics_path}")
    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED — INSTALLATION & INFERENCE READY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
