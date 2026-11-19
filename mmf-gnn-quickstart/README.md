# MMF-GNN: Multi-Modal Fusion Graph Neural Network

A multi-modal graph and sequence neural network for predicting anti-schistosome activity of small molecules from SMILES representations.

## Folder Contents

All training, inference, verification, and sample data live in this single folder:

- `mmf_gnn.py` — Shared `MMF_GNN` architecture (MSGA + EAGT + BiLSTM + Multi-Scale CNN + Bidirectional Cross-Modal Attention), RDKit graph featurization (42-dim node, 10-dim edge), SMILES encoders, evaluation, and `predict_smiles()` utilities.
- `quick_start.py` — End-to-end installation and inference pipeline verification script using `sample_dataset.csv`.
- `predict.py` — Standalone inference CLI for single SMILES strings or batch CSV screening.
- `train.py` — Full training pipeline with dataset merging, RDKit cleaning, class balancing, early stopping, checkpoint saving, and 10 publication-ready plots.
- `sample_dataset.csv` — Balanced 40-compound sample dataset (20 Active, 20 Inactive) for rapid verification without downloading the full dataset.
- `requirements.txt` — Python package dependencies.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start Verification (No Full Dataset Required)

Run `quick_start.py` to verify all dependencies, RDKit molecular graph construction, forward pass shapes, 5-epoch smoke training, checkpoint reload, and end-to-end inference on `sample_dataset.csv`:

```bash
python quick_start.py
```

Expected outputs in `quickstart_output/`:
- `quickstart_model.pt` — Smoke-tested MMF-GNN checkpoint
- `quickstart_predictions.csv` — Holdout sample predictions (`SMILES`, `Probability`, `Predicted_Label`, `Predicted_Class`, `True_Label`)
- `quickstart_metrics.csv` — Holdout evaluation metrics (`ACC`, `F1`, `AUPRC`, `AUC`, `CK`, `MCC`)

To verify inference against an existing pre-trained checkpoint without running smoke training:

```bash
python quick_start.py --checkpoint outputs/mmf_gnn_model.pt --skip_train
```

## Running Inference (`predict.py`)

Single or multiple SMILES strings:

```bash
python predict.py --checkpoint quickstart_output/quickstart_model.pt \
  --smiles "OC(=O)CNC(=O)CCC1=CC=CC=C1" "CCC(C)C1=C(OC)N=CC=N1"
```

Batch CSV file:

```bash
python predict.py --checkpoint quickstart_output/quickstart_model.pt \
  --input_csv sample_dataset.csv \
  --output_csv predictions.csv
```

## Full Dataset Training (`train.py`)

```bash
python train.py --data_path path/to/Data_2.csv --output_dir ./outputs --epochs 100
```
