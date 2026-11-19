#!/usr/bin/env python3
"""
MMF-GNN Standalone Inference Script

Predict anti-schistosome activity from SMILES strings or a CSV file using
a trained MMF-GNN checkpoint.

Examples:
    python predict.py --checkpoint outputs/mmf_gnn_model.pt --smiles "OC(=O)CNC(=O)CCC1=CC=CC=C1"
    python predict.py --checkpoint quickstart_output/quickstart_model.pt --input_csv sample_dataset.csv --output_csv predictions.csv
"""

import os
import argparse
import torch
import pandas as pd

from mmf_gnn import load_model_checkpoint, predict_smiles


def main():
    parser = argparse.ArgumentParser(description="Run MMF-GNN inference on SMILES or CSV.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint.")
    parser.add_argument("--smiles", type=str, nargs="*", default=None, help="One or more SMILES strings.")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to input CSV with a 'SMILES' column.")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Path to save predictions CSV.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5).")
    args = parser.parse_args()

    if not args.smiles and not args.input_csv:
        parser.error("Provide either --smiles or --input_csv.")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model_checkpoint(args.checkpoint, device=device)

    if args.smiles:
        results = predict_smiles(model, args.smiles, device=device, threshold=args.threshold)
        for r in results:
            if r["Valid"]:
                print(f"SMILES: {r['SMILES']} -> {r['Predicted_Class']} (p={r['Probability']:.4f})")
            else:
                print(f"SMILES: {r['SMILES']} -> INVALID SMILES")

    if args.input_csv:
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        if "canonical_smiles" in df.columns and "SMILES" not in df.columns:
            df = df.rename(columns={"canonical_smiles": "SMILES"})
        if "SMILES" not in df.columns:
            raise ValueError("Input CSV must contain a 'SMILES' column.")

        results = predict_smiles(model, df["SMILES"].tolist(), device=device, threshold=args.threshold)
        out_df = pd.DataFrame(results)
        if "Activity" in df.columns:
            out_df["True_Label"] = df["Activity"].values
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(out_df)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
