# Anomaly Detection in C2 Beaconing Traffic (Federated Learning)

This repository investigates how privacy-preserving federated learning can improve Command-and-Control (C2) beaconing detection across heterogeneous network environments. The work combines extensive exploratory analysis of CTU-13 and UGR'16 NetFlow datasets, custom preprocessing/feature engineering pipelines, and centralized/federated training workflows.

## Project Goals

- Understand the statistical signatures of C2 beaconing in academic (CTU-13) and ISP-scale (UGR'16) traffic captures.
- Build reproducible preprocessing pipelines that extract periodicity, entropy, and aggregation features relevant to beaconing.
- Prototype centralized and federated training pipelines (Flwr) for classical ML models and neural networks while logging experiments with MLflow.
- Preserve data privacy by simulating non-IID clients and evaluating privacy-aware strategies (secure aggregation, differential privacy) in future iterations.

## Data & Exploratory Analysis

Exploratory studies are documented in `docs/CTU13_DATASET_EDA.tex` and `docs/UGR16_DATASET_EDA.tex`; companion figures and tables live under `docs/figures/` and `docs/tables/`.

### CTU-13 Highlights

- Sample of 2.8M bi-directional NetFlows with 40,961 botnet-labelled events; strong imbalance vs. background traffic.
- Scenario V42 concentrates ~57% of its flows as botnet, making it a natural “infected client” candidate for federated simulations.
- 230 distinct connection states after normalization, underscoring the need for robust categorical encoding.
- Botnet flows show shorter duration and lower packet/byte counts than background traffic, aligning with beaconing hypotheses.

### UGR'16 Highlights

- Reservoir-sampled subset of 2.8M flows (seed 42) to mirror CTU-13 scale; stored in Parquet with standardized column names.
- Background traffic dominates (96.7%), followed by `anomaly-sshscan` (3.0%) and `blacklist` (0.4%); total malicious indicator captures 3.3% of flows.
- Malicious flows have shorter duration (~3.2s vs. 4.7s) but send more forward packets, suggesting bursty, low-volume communication.
- Temporal analysis reveals attack bursts concentrated in specific windows, useful for defining client-specific activity peaks in federated setups.

## Repository Layout

- `docs/` – LaTeX reports, figures, and tables generated from exploratory analysis.
- `notebooks/` – EDA notebooks (`eda_ctu13.ipynb`, `eda_ugr16.ipynb`) used to regenerate report assets.
- `src/preprocessing/` – Dataset-specific preprocessors, advanced beaconing feature engineering, and CLI/Python APIs (see `src/preprocessing/README.md`).
- `src/models/` – Implementations for neural networks, random forest, and XGBoost baselines.
- `src/federated/` – Client/server strategy definitions leveraging Flower (`flwr`).
- `src/scripts/` – Entry-points for preprocessing, cross-validation, and training in centralized or federated modes.
- `configs/` – YAML presets covering centralized and federated experiments with class re-balancing options.

## Getting Started

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

### Creating Dataset Samples

Before training, you need to create properly-sampled parquet files from the raw datasets:

```bash
# Quick start (uses paths from .env)
./create_samples.sh

# Or use Python directly
python src/scripts/create_dataset_samples.py
```

This creates stratified samples (~2.8M rows each) with proper representation of malicious traffic:

- `data/raw/ctu13_sample.parquet` - CTU-13 botnet detection dataset
- `data/raw/ugr16_sample.parquet` - UGR'16 ISP traffic with attacks

For detailed instructions, see:

- **Quick Reference**: [QUICK_SAMPLING_GUIDE.md](QUICK_SAMPLING_GUIDE.md)
- **Full Documentation**: [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md)

### Environment Variables

- Supply dataset paths and MLflow tracking URLs via a `.env` file (see `python-dotenv` dependency).
- GPU training is optional; Torch/Torchvision are included for deep learning baselines.

## Reproducing the EDA

1. Run `pdflatex` (twice) on `docs/CTU13_DATASET_EDA.tex` or `docs/UGR16_DATASET_EDA.tex` to compile the reports:pdflatex CTU13_DATASET_EDA.tex

   ```bash
   pdflatex CTU13_DATASET_EDA.tex
   ```

   Generated PDFs summarize dataset methodology, label distributions, temporal patterns, and implications for federated training.
2. Download or generate the processed Parquet/CSV inputs referenced in `data/raw/` and `data/processed/metadata` JSON files.
3. Launch `notebooks/eda_ctu13.ipynb` or `notebooks/eda_ugr16.ipynb` and execute sequentially to regenerate plots/tables.

## Training Workflows

- **Preprocessing:**

  ```bash
  python src/scripts/preprocess_ctu13.py --input data/raw/ctu13_sample.parquet --output data/processed --advanced
  python src/scripts/preprocess_ugr16.py --input data/raw/ugr16_sample.parquet --output data/processed --advanced
  ```
- **Centralized training:** `python src/scripts/train_centralized.py --config configs/centralized_config.yaml`
- **Cross-validation experiments:** `python src/scripts/cross_validate.py --config configs/cv_xgboost_balanced.yaml`
- **Federated simulation:** `python src/scripts/train_federated.py --config configs/federated_config.yaml`
- [ ] All runs log metrics and artifacts through MLflow (`src/utils/mlflow_logger.py`). Adjust class-balancing strategies with the provided config variants (class weights, balanced sampling, etc.).

## Roadmap

- Extend client partitioning strategies to combine CTU-13 scenario splits with UGR'16 ISP profiles.
- Integrate temporal beaconing metrics (FFT-based periodicity, jitter, entropy) into federated pipelines.
- Evaluate secure aggregation and differential privacy noise mechanisms on top of Flower strategies.
- Benchmark model families (Random Forest, XGBoost, MLP/LSTM) under non-IID client distributions using shared evaluation metrics from `src/evaluation/metrics.py`.

For additional details, consult the LaTeX reports in `docs/` and inline module documentation throughout `src/`.
