# Preprocessing Module

This module provides comprehensive preprocessing pipelines for the CTU-13 and UGR'16 datasets, designed specifically for C2 beaconing anomaly detection in federated learning scenarios.

## Features

### Base Preprocessing
- Data loading from parquet files
- Missing value handling
- Data type conversions
- Stratified train/validation/test splitting
- Metadata tracking

### Dataset-Specific Processing

#### CTU-13
- Hex port number conversion
- Binary label creation (botnet vs background)
- Scenario extraction
- Protocol and state encoding
- Temporal feature extraction

#### UGR'16
- Column name standardization
- Malicious traffic detection via keyword matching
- Packet direction features
- Advanced port and protocol encoding

### Advanced Feature Engineering

For C2 beaconing detection, the module includes specialized features:

#### Periodicity Features
- Inter-arrival time statistics
- Coefficient of variation (regularity indicator)
- Periodicity scores

#### Flow Aggregation Features
- Time-windowed flow counts
- Statistical summaries per source

#### Entropy Features
- Destination port entropy
- Protocol entropy
- Low entropy indicates focused communication

#### Size Consistency Features
- Packet and byte size variability
- Consistency scores (low CV = beaconing)

## Usage

### Command Line

#### Basic preprocessing:
```bash
# CTU-13
python src/scripts/preprocess_ctu13.py \
    --input data/raw/ctu13_sample.parquet \
    --output data/processed

# UGR'16
python src/scripts/preprocess_ugr16.py \
    --input data/raw/ugr16_sample.parquet \
    --output data/processed
```

#### Advanced preprocessing with C2 beaconing features:
```bash
# CTU-13
python src/scripts/preprocess_ctu13.py \
    --input data/raw/ctu13_sample.parquet \
    --output data/processed \
    --advanced \
    --test-size 0.2 \
    --val-size 0.1 \
    --random-seed 42

# UGR'16
python src/scripts/preprocess_ugr16.py \
    --input data/raw/ugr16_sample.parquet \
    --output data/processed \
    --advanced \
    --test-size 0.2 \
    --val-size 0.1 \
    --random-seed 42
```

### Python API

```python
from pathlib import Path
from preprocessing import CTU13Preprocessor, FeatureEngineer

# Initialize preprocessor
preprocessor = CTU13Preprocessor(
    input_path=Path('data/raw/ctu13_sample.parquet'),
    output_dir=Path('data/processed'),
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

# Run full preprocessing pipeline
train_df, val_df, test_df = preprocessor.process()

# Or run step by step with advanced features
preprocessor.load_data()
preprocessor.clean_data()
preprocessor.create_labels()
preprocessor.extract_features()

# Apply advanced C2 beaconing feature engineering
engineer = FeatureEngineer(time_window='1H')
preprocessor.df = engineer.engineer_features(
    preprocessor.df,
    timestamp_col='StartTime',
    enable_periodicity=True,
    enable_aggregation=True,
    enable_entropy=True,
    enable_consistency=True
)

# Finish preprocessing
preprocessor.handle_missing_values()
train_df, val_df, test_df = preprocessor.split_data(preprocessor.df)
preprocessor.save_processed_data(train_df, val_df, test_df)
```

## Output Structure

```
data/processed/
├── ctu13_train.parquet          # Training set (basic features)
├── ctu13_val.parquet            # Validation set
├── ctu13_test.parquet           # Test set
├── ctu13_metadata.json          # Preprocessing metadata
├── ctu13_advanced_train.parquet # Training set (advanced features)
├── ctu13_advanced_val.parquet
├── ctu13_advanced_test.parquet
├── ctu13_advanced_metadata.json
├── ugr16_train.parquet
├── ugr16_val.parquet
├── ugr16_test.parquet
├── ugr16_metadata.json
├── ugr16_advanced_train.parquet
├── ugr16_advanced_val.parquet
├── ugr16_advanced_test.parquet
└── ugr16_advanced_metadata.json
```

## Metadata

Each preprocessing run generates a metadata JSON file containing:
- Sample counts (train/val/test)
- Feature list
- Label distribution
- Imbalance ratio
- Processing parameters
- Feature engineering steps applied

## C2 Beaconing Detection

The advanced feature engineering is specifically designed for detecting Command-and-Control beaconing patterns, which typically exhibit:

1. **Periodicity**: Regular intervals between communications
2. **Low Entropy**: Focused destination patterns
3. **Consistent Sizes**: Similar packet/byte counts
4. **Regular Flow Counts**: Predictable communication frequency

These behavioral signatures distinguish C2 beaconing from normal network traffic and other types of attacks.

## Requirements

- pandas >= 2.0
- numpy >= 1.20
- scikit-learn >= 1.0
- scipy >= 1.7
- fastparquet >= 0.8

## Integration with Federated Learning

The preprocessed data is designed for federated learning scenarios:
- Consistent feature engineering across datasets
- Stratified splits maintain class distribution
- Parquet format for efficient I/O
- Metadata enables reproducibility
- Support for partitioning across clients (future work)
