# Enhanced Graph Anomaly Detection

Enhanced graph neural network models for anomaly detection with neighborhood similarity improvements.

## Models

### Baseline Models

- `gcn`: Graph Convolutional Network
- `gat`: Graph Attention Network
- `graphsage`: GraphSAGE

### Enhanced Models

- `ns-gcn`: Neighborhood Similarity GCN
- `ns-gat`: Neighborhood Similarity GAT
- `ns-sage`: Neighborhood Similarity GraphSAGE

### Adaptive Models

- `ans-gcn`: Adaptive Neighborhood Similarity GCN
- `ans-gat`: Adaptive Neighborhood Similarity GAT
- `ans-sage`: Adaptive Neighborhood Similarity GraphSAGE

## Usage

### Simple Usage

```bash
# Run single model
python run_experiment_simple.py --model ns-gcn --dataset tolokers --trials 5

# Run multiple models
python run_experiment_simple.py --model gcn,ns-gcn,ans-gcn --dataset tolokers --trials 3

# Run all models
python run_experiment_simple.py --model all --dataset all --trials 1
```

### Enhanced Usage (with advanced results management)

```bash
# Run with enhanced results tracking
python run_experiment_enhanced.py --model ns-gat --dataset tolokers --trials 5

# Run with custom auto-save interval
python run_experiment_enhanced.py --model all --dataset all --trials 3 --auto-save-interval 60

# Run multiple experiments with automatic organization
python run_experiment_enhanced.py --model ns-gcn,ans-gcn --dataset tolokers,reddit --trials 5
```

### Results Management

```bash
# List all experiments
python view_results.py --list

# View specific experiment details
python view_results.py --view ns-gcn_tolokers_trials5_20241224_143022

# Compare multiple experiments
python view_results.py --compare exp1_id exp2_id exp3_id

# Export results to CSV
python view_results.py --export exp_id --output results.csv

# Clean up old experiments (older than 30 days)
python view_results.py --cleanup 30
```

### Full Usage (original with basic features)

```bash
python run_experiment.py --model ns-gat --dataset tolokers --trials 5
```

## Datasets

- `reddit`: Reddit social network
- `weibo`: Weibo social network
- `tolokers`: Tolokers crowdsourcing
- `questions`: Questions dataset

## Core Innovation

Enhanced models promote 3-hop neighbors to 2-hop based on:

- Node similarity computation
- Degree-aware enhancement strategy
- Adaptive parameter adjustment (for ans-\* models)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── baseline_models.py      # GCN, GAT, GraphSAGE
│   │   ├── base_enhanced.py        # Shared enhancement logic
│   │   ├── enhanced_gcn.py         # Enhanced GCN models
│   │   ├── enhanced_gat.py         # Enhanced GAT models
│   │   └── enhanced_graphsage.py   # Enhanced GraphSAGE models
│   ├── data/                       # Data loading utilities
│   ├── training/                   # Training utilities
│   └── utils/                      # Utility functions
├── datasets/                       # Dataset files
├── scripts/                        # Utility scripts
├── results/                        # Experiment results (auto-generated)
├── config.py                       # Configuration
├── run_experiment.py               # Original experiment runner
├── run_experiment_simple.py        # Simplified experiment runner
├── run_experiment_enhanced.py      # Enhanced experiment runner with advanced results
└── view_results.py                 # Results viewer and analysis tool
```
