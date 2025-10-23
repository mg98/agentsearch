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


def load_data_graph(data_dir, percentage_lying: int=0, incl_lying_label: bool=False):
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

    label_fp = "/mnt/lourens/data/agentsearch/data/graph/corrupt_agents.json"
    import json
    with open(label_fp, 'r') as f:
        labels = json.load(f)[str(percentage_lying)]

    agents_df['is_corrupt'] = 0
    for agent_id in labels:
        if agent_id in agents_df.index:
            agents_df.at[agent_id, 'is_corrupt'] = 1
        else:
            raise ValueError(f"Agent id {agent_id} in corrupt_agents.json not found in agents.parquet")

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
        x_corrupt = torch.tensor(
            [agents_df.loc[agent_id]['is_corrupt'] for agent_id in agent_ids]
        ),
        edge_index = torch.tensor( edge_df[['source_agent', 'target_agent']].astype(int).values.T),
        edge_attr = torch.stack([torch.tensor(t).float() for t in edge_df['question_emb'].values.tolist()]),
        y = torch.tensor(edge_df['binary_score'].values.astype(float).tolist())
    )

    return data    


def load_data_tab(data_dir, percentage_lying: int=0):
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

    edge_df['agent_emb'] = edge_df['target_agent'].map(agents_df['embedding'])
    edge_df['feat'] = edge_df.apply(lambda row: torch.cat((torch.tensor(row['question_emb']), torch.tensor(row['agent_emb']))), axis=1)

    X = torch.stack(edge_df['feat'].to_list())
    y = torch.tensor(edge_df['binary_score'].to_numpy()).long()
    return X, y

def load_data_transformer(data_dir, percentage_lying: int=0, with_asker_id=False):
    """Load data from parquet files"""
    avail_edge_files = os.listdir(os.path.join(data_dir, "graph"))
    assert f"edges_{percentage_lying}.csv" in avail_edge_files

    edge_df = pd.read_csv(os.path.join(data_dir, "graph", f"edges_{percentage_lying}.csv"))
    edge_df['binary_score'] = edge_df['score'] > 0  # binary score can be derived from the score
    
    print(f"Mean binary score: {edge_df['binary_score'].mean():.4f}")

    questions_df = pd.read_parquet(os.path.join(data_dir, "questions.parquet")).set_index('id')
    edge_df['question_emb'] = edge_df['question'].map(questions_df['embedding'])
    # convert to list

    agents_df = pd.read_parquet(os.path.join(data_dir, "agents.parquet")).set_index('id')
    agent_ids = pd.Index(edge_df['target_agent'].tolist()).unique()
    agents = torch.stack(
        [
            torch.tensor(agents_df.loc[agent_id]['embedding']).float() for agent_id in agent_ids
        ]
    )

    if with_asker_id:
        edge_df['question_emb_and_score'] = edge_df.apply(
            lambda row: torch.cat(
                (
                    torch.tensor([row['source_agent']]),
                    torch.tensor(row['question_emb']), 
                    torch.tensor([row['binary_score']])
                ), 
                dim=0
            ), 
            axis=1
        )
    else:
        edge_df['question_emb_and_score'] = edge_df.apply(
            lambda row: torch.cat(
                (
                    torch.tensor(row['question_emb']), 
                    torch.tensor([row['binary_score']])
                ), 
                dim=0
            ), 
            axis=1
        )

    # TODO use padding here
    # dictionary mapping agent_id to tensor with all questions asked to that agent
    questions_by_agent = [
        torch.stack(
            edge_df[edge_df['target_agent'] == agent_id]['question_emb_and_score'].tolist()
        )
        for agent_id in agent_ids
    ]

    return questions_by_agent, agents

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
    graph_data = load_data_graph(config['data']['root'], percentage_lying=config['data']['percentage_lying'])
    
    print(f"Graph statistics:")
    print(f"  Nodes: {graph_data.x.size(0)}, embedding size {graph_data.x.size(1)}")
    print(f"  Edges: {graph_data.edge_index.size(1)}")
    print(f"  Node features: {graph_data.x.size(1)}") 
    print(f"  Edge features: {graph_data.edge_attr.size(1)}")
    
    # Prepare data (splits and normalization)
    print("Preparing data splits and normalization...")
    data, split_dict, device = prepare_data(graph_data, config)

    # assert no overlap between train, val, test masks
    assert (split_dict['train_mask'] & split_dict['val_mask']).sum() == 0, "Train and val masks should not overlap."
    assert (split_dict['train_mask'] & split_dict['test_mask']).sum() == 0, "Train and test masks should not overlap."
    assert (split_dict['val_mask'] & split_dict['test_mask']).sum() == 0, "Val and test masks should not overlap."

    
    # Setup model
    print("Setting up model...")
    model, head = setup_model_and_training(graph_data, config)
    model = model.to(device)
    head = head.to(device)
    
    print(f"Training on device: {device}")
    
    # Train model
    logging.info("Starting training...")
    test_metrics = train_model(model, head, data, split_dict, config)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Final Test Results:")
    for k, v in test_metrics.items():
        logging.info(f"  {k}: {v:.6f}")
    print("="*50)


def main_mlp(perc_lying: int = 0):
    """Main function to test MLP"""
    from gnnexp.mlp import MLP, train_model

    # Paths
    # TODO make --cfg
    config_path = "./configs/config_mlp.yaml"
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)

    X, y = load_data_tab(config['data']['root'], percentage_lying=perc_lying)

    device = config['experiment']['device']
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    X = X.float().to(device)
    y = y.float().to(device)

    model = MLP(X.shape[1], config['model']['hidden_dim'], 1, config['model']['num_layers'], config['model']['dropout'])
    model.to(device)

    return train_model(model, X, y, config)


def main_transformer(perc_lying: int = None):
    """Main function to test Transformer-based model"""
    from gnnexp.transformer import TabularPredictor, train_model

    # Paths
    # TODO make --cfg
    config_path = "./configs/config_transformer.yaml"
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)

    if perc_lying is None:
        perc_lying = config['data']['percentage_lying']
    else:
        config['data']['percentage_lying'] = perc_lying

    questions_by_agent, agent = load_data_transformer(
        config['data']['root'], 
        percentage_lying=perc_lying, 
        with_asker_id=config['model'].get('add_asker_id', False)
    )

    assert len(questions_by_agent) == agent.shape[0], "Number of agents mismatch between questions and agent embeddings"

    device = config['experiment']['device']
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # set to device
    agents = agent.float().to(device)
    for t in questions_by_agent:
        t = t.float().to(device)

    model = TabularPredictor(
        questions_by_agent[0].shape[1] - 1, 
        agent.shape[1], 
        config['model']['hidden_dim'],
        config['model'].get('n_heads', 4),
        config['model'].get('n_layers', 3)
        )
    model.to(device)

    return train_model(model, questions_by_agent, agents, config)
    


if __name__ == "__main__":
    # Set logging to INFO level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # main_mlp()
    main_transformer()

