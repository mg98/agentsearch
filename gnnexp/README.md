# Edge Classification with Graph Neural Networks

This project implements an edge classification system using Graph Neural Networks (GNNs) for predicting trust scores on a graph dataset.

## Overview

The system includes:
- **Node Embedder**: MLP-based node feature embedder
- **Edge Embedder**: MLP-based edge feature embedder  
- **Message Passing**: Graph Convolutional Network (GCN) for message passing
- **Prediction Head**: MLP for final edge prediction
- **Train/Val/Test Splits**: Automatic edge splitting with logarithmic normalization
- **Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Reproducibility**: Seeded random number generation

## Files Structure

```
├── config.yaml          # Configuration file
├── models.py            # Model architectures
├── data_utils.py        # Data preprocessing and splitting utilities
├── training.py          # Training and evaluation functions
├── main_train.py        # Main training script
├── evaluate.py          # Evaluation script for trained models
├── demo.py             # Quick demo script
└── models/             # Directory for saved models
```

## Quick Start

### 1. Run the Demo

To quickly test the system with reduced epochs:

```bash
python demo.py
```

### 2. Full Training

To run full training with the configuration in `config.yaml`:

```bash
python main_train.py
```

### 3. Evaluate a Trained Model

```bash
python evaluate.py --model_path models/edge_classification_best.pth
```

## Configuration

The `config.yaml` file contains all configuration parameters:

### Model Architecture
```yaml
model:
  node_embedding_dim: 128      # Node embedding dimension
  edge_embedding_dim: 64       # Edge embedding dimension
  hidden_dim: 256             # Hidden layer dimension
  num_layers: 3               # Number of GCN layers
  dropout: 0.1                # Dropout rate
```

### Training Parameters
```yaml
training:
  epochs: 200                 # Maximum training epochs
  learning_rate: 0.001        # Learning rate
  weight_decay: 1e-5          # L2 regularization
  patience: 20                # Early stopping patience
  min_delta: 1e-4            # Minimum improvement for early stopping
```

### Data Configuration
```yaml
data:
  train_ratio: 0.7           # Training split ratio
  val_ratio: 0.15            # Validation split ratio  
  test_ratio: 0.15           # Test split ratio
  log_normalize: true        # Apply logarithmic normalization
```

### Experiment Settings
```yaml
experiment:
  seed: 42                   # Random seed for reproducibility
  device: "cuda"             # Device to use (cuda/cpu)
  project_name: "agentsearch" # Wandb project name
  experiment_name: "edge_classification"
  save_model: true           # Save best model
```

## Model Architecture

### Node Embedder
- Input: Node features (776-dimensional)
- Output: Node embeddings (128-dimensional)
- Architecture: MLP with ReLU activation and dropout

### Edge Embedder  
- Input: Edge features (769-dimensional)
- Output: Edge embeddings (64-dimensional)
- Architecture: MLP with ReLU activation and dropout

### Message Passing
- Input: Node embeddings
- Output: Updated node embeddings
- Architecture: Multi-layer GCN with ReLU activation

### Prediction Head
- Input: Source node embedding + Target node embedding + Edge embedding
- Output: Edge prediction (scalar)
- Architecture: MLP with ReLU activation and dropout

## Data Processing

### Edge Splitting
- Edges are randomly split into train/val/test sets
- Split ratios are configurable
- Splits maintain the same random seed for reproducibility

### Logarithmic Normalization
- Trust scores are log-transformed: `log(score + ε)`
- Normalized within each split to have mean=0, std=1
- Applied separately to train/val/test sets

## Training Process

### Loss Function
- Mean Squared Error (MSE) for regression

### Optimizer
- Adam optimizer with configurable learning rate and weight decay

### Early Stopping
- Monitors validation loss
- Configurable patience and minimum delta
- Saves best model state based on validation performance

### Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error
- **Correlation**: Pearson correlation coefficient

## Experiment Tracking

The system integrates with Weights & Biases (wandb) for experiment tracking:

- Automatic logging of training/validation metrics
- Model artifact saving
- Configurable project and experiment names
- Real-time training progress monitoring

### Setup Wandb

```bash
# Login to wandb (first time only)
wandb login

# Or set environment variable
export WANDB_API_KEY=your_api_key
```

## Reproducibility

All random operations are seeded using the configured seed:
- PyTorch random number generation
- NumPy random number generation  
- CUDA random number generation
- Deterministic CUDA operations

## Usage Examples

### Custom Configuration

Create a new config file or modify `config.yaml`:

```python
import yaml

# Load and modify config
config = yaml.safe_load(open('config.yaml', 'r'))
config['training']['learning_rate'] = 0.01
config['model']['hidden_dim'] = 512

# Save modified config
with open('custom_config.yaml', 'w') as f:
    yaml.dump(config, f)
```

### Programmatic Usage

```python
from models import EdgeClassificationModel
from data_utils import prepare_data
from training import setup_model_and_training, train_model

# Load your data and config
graph_data = load_your_data()
config = load_your_config()

# Prepare data
data, split_dict, device = prepare_data(graph_data, config)

# Setup and train model
model = setup_model_and_training(graph_data, config)
model = model.to(device)
test_metrics = train_model(model, data, split_dict, config, device)
```

### Making Predictions

```python
from evaluate import predict_edges

# Load trained model and make predictions
predictions = predict_edges(
    model_path='models/edge_classification_best.pth',
    graph_data=your_graph_data,
    edge_indices=[0, 1, 2, 100]  # Specific edge indices
)
```

## Performance Tips

1. **GPU Usage**: Set `device: "cuda"` in config for GPU acceleration
2. **Batch Size**: The system loads the full graph (no batching needed for small graphs)
3. **Memory**: Monitor GPU memory usage for large graphs
4. **Early Stopping**: Adjust patience based on dataset size and complexity

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce model dimensions or use CPU
2. **Wandb login required**: Run `wandb login` or set API key
3. **Import errors**: Ensure all dependencies are installed

### Dependencies

The system requires:
- PyTorch
- PyTorch Geometric
- scikit-learn
- wandb
- PyYAML
- numpy
- tqdm

## Contributing

To extend the system:
1. Add new model architectures in `models.py`
2. Add new metrics in `training.py`
3. Modify data preprocessing in `data_utils.py`
4. Update configuration schema in `config.yaml`
