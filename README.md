# MMF-GNN: Multimodal Fusion Graph Neural Network Architecture for Prediction of Active Small Molecules Against Erm (41) of Mycobacterium abscessus

A deep learning model for predicting molecular activity using graph neural networks.

---

## Installation

```bash
pip install torch torch-geometric rdkit numpy pandas scikit-learn matplotlib seaborn
```

---

## Training

```bash
python train.py \
    --data_path /path/to/Data_1.csv /path/to/Data_2.csv /path/to/Data_3.csv \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --gpu 0
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | Required | Path(s) to CSV file(s) |
| `--output_dir` | `./output` | Output directory |
| `--epochs` | `100` | Number of epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--dropout` | `0.3` | Dropout rate |
| `--hidden_dim` | `128` | Hidden dimension |
| `--num_heads` | `4` | Attention heads |
| `--test_size` | `0.2` | Test split ratio |
| `--gpu` | `0` | GPU ID |

### Training Output

```
output/
├── mmf_gnn_model.pt          # Trained model
├── train4results.csv         # Metrics (ACC, F1, AUPRC, AUC, CK, MCC)
├── train4predictions.csv     # Test predictions
└── *.png                     # Training plots
```

---

## Testing / Inference

```bash
python test.py \
    --model_path ./output/mmf_gnn_model.pt \
    --data_path /path/to/test_data.csv \
    --output_dir ./predictions \
    --gpu 0
```

### Test Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to trained model (.pt) |
| `--data_path` | Required | Path to test CSV |
| `--output_dir` | `./predictions` | Output directory |
| `--batch_size` | `16` | Batch size |
| `--threshold` | `0.5` | Classification threshold |
| `--gpu` | `0` | GPU ID |

### Test Output

```
predictions/
├── predictions.csv           # SMILES, Probability, Predicted_Label
├── test_metrics.csv          # Metrics (if labels provided)
└── *.png                     # Evaluation plots
```

---

## Data Format

CSV file with columns:

```csv
SMILES,Activity
CCO,0
CC(=O)OC1=CC=CC=C1C(=O)O,1
```

- `SMILES`: Molecule SMILES string
- `Activity`: 0 (Inactive) or 1 (Active)

---

## License

MIT License
