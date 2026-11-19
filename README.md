# Multi-GNN for Privacy-Preserving Machine Learning (PPML)

This repository contains an implementation of Multi-GNN adapted for privacy-preserving machine learning (PPML) over secure Multi-Party Computation (MPC). The base code builds upon directed multigraph GNNs for anti-money laundering (AML) transaction networks, modified to eliminate communication bottlenecks that make standard GNN training impractical in MPC.

Specifically, we replace communication-heavy primitives with zero-round MPC equivalents:
1. **Scheduled Zero Round Normalization (`ScheduledZeroRoundNorm`)** instead of Batch Normalization.
2. **SGD with Momentum** instead of Adam.

---

## Why These Modifications?

Training neural networks inside secret-sharing MPC protocols is bounded by communication latency (network rounds) and total volume of opened secret shares:

- **Batch Normalization to Zero Round Norm:** Standard BatchNorm requires computing batch variances and reciprocal square roots via iterative numerical methods (such as Newton-Raphson), requiring around 15 communication rounds per layer. `ScheduledZeroRoundNorm` centers features by mean subtraction and applies a scheduled, public scalar multiplier based on the epoch ($\gamma_t = 1.0 + \ln(1.0 + t \cdot \text{growth\_rate})$). Because this scaling factor is public, it requires **0 communication rounds** and opens **0 secret shares**.
- **Adam to SGD with Momentum:** Adam requires tracking first and second moments per parameter, followed by division and square root operations ($\approx 17$ rounds per parameter update). Replacing Adam with SGD with momentum ($\mu = 0.9$) moves all optimizer updates to local linear operations on shares, dropping the optimizer communication cost to **0 rounds**.
- **MPC Telemetry Tracker:** `training.py` includes an `MPCTelemetry` class to track the exact number of communication rounds, opened secret elements, estimated network bandwidth, and local MAC operations using Matrix Beaver Triples across training epochs.

---

## Project Structure

```text
Multi-GNN/
├── checkpoints/                 # Reference scripts and ablation checkpoints
│   ├── final-submission/        # Latest training, model, and data loading scripts
│   ├── MPC_BN_SGD/              # Experiments testing BatchNorm with SGD under MPC
│   ├── No_Batch_Norm/           # Ablation scripts without normalization layers
│   └── No_ReLU/                 # Ablation scripts experimenting with polynomial activations (PolyAct)
├── simulation/                  # MPC cost simulation scripts and comparison plots
│   ├── relu-SGD-beaver-matrices/# Matrix Beaver Triple cost models and 100-epoch comparison plots
│   └── relu-SGD-beaver-scalars/ # Scalar Beaver Triple cost models and comparison plots
├── logs/                        # Standard logging directory
│   └── logs.log                 # Generated automatically on runs via logger_setup()
├── test_logs/                   # Output logs from experimental runs (SGD tests, Zero Round Norm, etc.)
├── wandb/                       # Local Weights & Biases telemetry and run artifacts
├── command.txt                  # Sample nohup command for running in the background on GPU
├── data_config.json             # Paths to data and saved models (must be edited on setup)
├── data_loading.py              # AML dataset parsing, feature engineering, and temporal data splits
├── data_util.py                 # Graph data structures, z-normalization, and graph transforms
├── env.yml                      # Conda environment definition
├── format_kaggle_files.py       # Preprocesses raw Kaggle transaction files into formatted_transactions.csv
├── inference.py                 # Evaluates a saved model checkpoint
├── main.py                      # Main entrypoint for training and inference
├── model_settings.json          # Architecture configurations and hyperparameter settings
├── models.py                    # GNN architectures (GINe, GATe, PNA, RGCN, ScheduledZeroRoundNorm)
├── train_util.py                # DataLoader batching, seed edge tracking, ego IDs, and checkpointing
├── training.py                  # Training loop, validation/test metrics, and MPC telemetry logging
└── util.py                      # CLI argument parser, random seed setter, and logging setup
```

---

## Setup

Create the Conda environment from `env.yml` and activate it:

```bash
conda env create -f env.yml
conda activate multignn
```

Key dependencies: Python 3.9, PyTorch, PyTorch Geometric (PyG), datatable, pandas, scikit-learn, and wandb.

---

## Data Download and Preparation

This project uses the **IBM Transactions for Anti-Money Laundering (AML)** dataset.

1. **Download:** Grab the dataset from Kaggle:  
   https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data  
   (Download transaction files such as `HI-Small_Trans.csv` or `HI-Medium_Trans.csv`).

2. **Preprocess:** Run `format_kaggle_files.py` on the raw CSV:
   ```bash
   python format_kaggle_files.py /path/to/HI-Small_Trans.csv
   ```
   This produces a `formatted_transactions.csv` file in the same directory.

3. **Directory Layout:** Place `formatted_transactions.csv` in a subfolder named after your dataset argument (e.g. `Small_HI` or `data`):
   ```text
   /path/to/aml_data/
   └── Small_HI/
       └── formatted_transactions.csv
   ```

---

## Files to Change When Cloning

When setting this up on a new machine or workspace, check and update these files:

### 1. `data_config.json`
Point the paths to your local data folder and checkpoint directories:
```json
{
  "paths": {
    "aml_data": "/path/to/aml_data",
    "model_to_load": "/path/to/saved_models",
    "model_to_save": "/path/to/save_models"
  }
}
```
`data_loading.py` looks for:
```python
f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions.csv"
```
So if you run `--data Small_HI`, it reads `/path/to/aml_data/Small_HI/formatted_transactions.csv`.

### 2. `training.py`
If you track runs using Weights & Biases, update the project name in `wandb.init()`:
```python
wandb.init(
    mode="disabled" if args.testing else "online",
    project="your_project_name",
    ...
)
```
If you do not want to use W&B, pass the `--testing` argument when running.

### 3. Hyperparameters & Environment Variables (Optional)
- Model architecture defaults can be tweaked in `model_settings.json`.
- Optimizer values can also be overridden through environment variables without modifying source code:
  - `ADAM_LR`: sets the SGD learning rate (variable name preserved for compatibility, defaults to `extract_param("lr", args)`).
  - `MOMENTUM`: momentum factor for SGD (default: `0.9`).
  - `W_CE2`: minority class weight for cross-entropy loss (default from `model_settings.json`).

---

## Running the Code

### Training GIN with Zero Round Norm and SGD
```bash
python main.py \
    --data Small_HI \
    --model gin \
    --batch_size 1024 \
    --tqdm \
    --tds \
    --n_epochs 50 \
    --unique_name
```

### Running in the Background on GPU
See `command.txt` for an example of launching a detached background run:
```bash
nohup env CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --data Small_HI \
    --model gin \
    --batch_size 1024 \
    --tqdm \
    --tds \
    --n_epochs 50 \
    --unique_name > test_logs/my_run.log 2>&1 &
```

### Supported Models (`--model`)
- `gin`: GIN with edge features (`GINe`) using Scheduled Zero Round Normalization.
- `gat`: Graph Attention Network with edge features (`GATe`).
- `pna`: Principal Neighbourhood Aggregation (`PNA`).
- `rgcn`: Relational Graph Convolutional Network (`RGCN`).

### Useful Arguments
- `--data <folder>`: Dataset directory name under `aml_data` (required).
- `--model <arch>`: Architecture to run (`gin`, `gat`, `pna`, `rgcn`) (required).
- `--batch_size <int>`: Batch size (default: `8192`).
- `--n_epochs <int>`: Epoch count (default: `100`).
- `--tds`: Enables time-delta edge features between transactions.
- `--ports`: Enables port numbering features.
- `--ego`: Adds ego IDs to center nodes in sampled batches.
- `--emlps`: Enables edge MLP updates.
- `--save_model`: Saves the best model checkpoint to `model_to_save`.
- `--inference`: Runs inference using the model from `model_to_load`.
- `--finetune`: Loads model checkpoint from `model_to_load` and continues training.
- `--testing`: Disables W&B logging.
- `--tqdm`: Enables the progress bar.

---

## Logs and MPC Simulation

- **Logs:**
  - Runtime logs are written to `logs/logs.log` by default.
  - Test run outputs and comparison logs from prior experiments live in `test_logs/`.
- **MPC Telemetry Simulator:**
  To run the analytical communication rounds and bandwidth comparison between baseline Adam/BatchNorm and our Zero Round Norm/SGD setup:
  ```bash
  cd simulation/relu-SGD-beaver-matrices
  python mpc_comparison_simulator.py
  ```
  This generates comparison plots (such as `mpc_100_epoch_matrix_triples.png`) showing communication rounds, opened elements, and total communication volume across training epochs.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
