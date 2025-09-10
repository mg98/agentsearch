import os
import sys
import yaml
import logging
import re
import torch
from torch_geometric.data import Data

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gnnexp.data_utils import prepare_data, set_seed
from gnnexp.training import setup_model_and_training, train_model

import pandas as pd

import torch
from torch_geometric.data import Data


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


def load_data(data_dir, percentage_lying: int=0):
    """Load data from parquet files"""
    avail_edge_files = os.listdir(os.path.join(data_dir, "graph"))
    assert f"edges_{percentage_lying}.csv" in avail_edge_files

    edge_df = pd.read_csv(os.path.join(data_dir, "graph", f"edges_{percentage_lying}.csv"))
    edge_df['binary_score'] = edge_df['score'] > 0  # binary score can be derived from the score
    

    questions_df = pd.read_parquet(os.path.join(data_dir, "questions.parquet")).set_index('id')
    edge_df['question_emb'] = edge_df['question'].map(questions_df['embedding'])

    agents_df = pd.read_parquet(os.path.join(data_dir, "agents.parquet")).set_index('id')
    # add a zero embedding for agent id 0
    agents_df.loc[0] = [torch.zeros(agents_df.iloc[0]['embedding'].shape[0]).numpy()]  # assuming all embeddings have the same shape

    # check that all edge_df['source_agent'] and edge_df['target_agent'] are in agents_df.index
    assert edge_df['source_agent'].isin(agents_df.index).all()
    assert edge_df['target_agent'].isin(agents_df.index).all()

    agent_ids = pd.Index(edge_df['source_agent'].tolist() + edge_df['target_agent'].tolist()).unique()
    map_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    edge_df['source_agent'] = edge_df['source_agent'].map(map_to_index)
    edge_df['target_agent'] = edge_df['target_agent'].map(map_to_index)

    data = Data(
        x = torch.stack(
            [
                torch.tensor(agents_df.loc[agent_id]['embedding']).float() for agent_id in agent_ids
            ]
        ),
        edge_index = torch.tensor( edge_df[['source_agent', 'target_agent']].astype(int).values.T),
        edge_attr = torch.stack([torch.tensor(t).float() for t in edge_df['question_emb'].values.tolist()]),
        y = torch.tensor(edge_df['binary_score'].values.astype(float).tolist())
    )

    return data    


def main():
    """Main training function"""
    
    # Paths
    # TODO make --cfg
    config_path = "./configs/config.yaml"
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Set seed early
    set_seed(config['experiment']['seed'])
    
    # Load data
    print("Loading graph data...")
    graph_data = load_data(config['data']['root'], percentage_lying=config['data']['percentage_lying'])
    
    print(f"Graph statistics:")
    print(f"  Nodes: {graph_data.x.size(0)}, embedding size {graph_data.x.size(1)}")
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
    print("="*50)


if __name__ == "__main__":
    # Set logging to INFO level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
