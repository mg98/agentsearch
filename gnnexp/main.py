import os
import sys
import yaml
import pickle
import logging
import re
from torch_geometric.data import Data

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils import prepare_data, set_seed
from gnnexp.training import setup_model_and_training, train_model


def convert_scientific_notation(obj):
    """
    Recursively convert scientific notation strings to floats in a nested data structure.
    This handles cases where YAML parsers might interpret scientific notation as strings.
    """
    if isinstance(obj, dict):
        return {key: convert_scientific_notation(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_scientific_notation(item) for item in obj]
    elif isinstance(obj, str):
        # Check if string matches scientific notation pattern
        scientific_pattern = r'^[-+]?[0-9]*\.?[0-9]+[eE][-+]?[0-9]+$'
        if re.match(scientific_pattern, obj):
            try:
                return float(obj)
            except ValueError:
                pass
    return obj


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert any scientific notation strings to floats
    config = convert_scientific_notation(config)
    
    return config


def load_graph_data(data_path):
    """Load graph data from pickle file"""
    with open(data_path, 'rb') as f:
        grph = pickle.load(f)['graph']
    
    # Create PyTorch Geometric Data object
    graph_data = Data(
        x=grph['x'], # shape [num_nodes, num_node_features]
        edge_index=grph['edge_index'], # shape [2, num_edges]
        edge_attr=grph['edge_attributes'][:, :-1], # shape [num_edges, num_edge_features]
        y=grph['trust_scores'] # shape [num_edges, 1] for edge regression
    )
    
    return graph_data


def main():
    """Main training function"""
    
    # Paths
    config_path = "./configs/config.yaml"
    data_path = "./data/graph.pkl"
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Set seed early
    set_seed(config['experiment']['seed'])
    
    # Load data
    print("Loading graph data...")
    graph_data = load_graph_data(data_path)
    
    print(f"Graph statistics:")
    print(f"  Nodes: {graph_data.x.size(0)}")
    print(f"  Edges: {graph_data.edge_index.size(1)}")
    print(f"  Node features: {graph_data.x.size(1)}")
    print(f"  Edge features: {graph_data.edge_attr.size(1)}")
    
    # Prepare data (splits and normalization)
    print("Preparing data splits and normalization...")
    data, split_dict, device = prepare_data(graph_data, config)
    
    # Setup model
    print("Setting up model...")
    model, head = setup_model_and_training(graph_data, config)
    model = model.to(device)
    head = head.to(device)
    
    print(f"Training on device: {device}")
    
    # Train model
    logging.info("Starting training...")
    test_metrics = train_model(model, head, data, split_dict, config, device)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Final Test Results:")
    print(f"  Loss: {test_metrics['loss']:.6f}")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  Correlation: {test_metrics['correlation']:.6f}")
    print("="*50)


if __name__ == "__main__":
    # Set logging to INFO level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
