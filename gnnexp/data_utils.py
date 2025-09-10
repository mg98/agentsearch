import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import logging


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_edge_splits(graph_data, train_ratio, val_ratio, test_ratio, seed):
    """
    Create train/val/test splits for edges and apply logarithmic normalization
    
    Args:
        graph_data: PyTorch Geometric Data object
        train_ratio: Fraction of edges for training
        val_ratio: Fraction of edges for validation
        test_ratio: Fraction of edges for testing
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing split data and masks
    """
    set_seed(seed)
    
    num_edges = graph_data.edge_index.size(1)
    edge_indices = np.arange(num_edges)
    
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        edge_indices, 
        test_size=test_ratio, 
        random_state=seed
    )
    
    # Second split: separate train and val from remaining edges
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size_adjusted,
        random_state=seed
    )
    
    # Create masks
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
        
    # assert that there is no data leakage
    assert (train_mask & val_mask).sum() == 0, "Train and validation masks should not overlap."
    assert (train_mask & test_mask).sum() == 0, "Train and test masks should not overlap."
    assert (val_mask & test_mask).sum() == 0, "Validation and test masks should not overlap."
    assert (train_mask | val_mask | test_mask).sum() == num_edges, "All edges should be covered by the masks."
    assert np.concatenate((train_indices, val_indices, test_indices)).size == num_edges, "All edge indices should be covered by the splits."
    assert np.unique(np.concatenate((train_indices, val_indices, test_indices))).size == num_edges, "No duplicate edge indices in splits."

    logging.info(f"Edge split statistics:")
    logging.info(f"  Train edges: {train_mask.sum().item()} ({train_mask.sum().item()/num_edges:.2%})")
    logging.info(f"  Val edges: {val_mask.sum().item()} ({val_mask.sum().item()/num_edges:.2%})")
    logging.info(f"  Test edges: {test_mask.sum().item()} ({test_mask.sum().item()/num_edges:.2%})")

    return {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }


def prepare_data(graph_data, config):
    """
    Prepare data for training including splits and normalization
    
    Args:
        graph_data: PyTorch Geometric Data object
        config: Configuration dictionary
    
    Returns:
        Prepared data dictionary
    """
    # Create edge splits
    split_dict = create_edge_splits(
        graph_data,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        seed=config['experiment']['seed']
    )
    
    # Move to device if specified
    device = config['experiment']['device']
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Create data object 
    data = Data(
        x=graph_data.x.to(device),
        edge_index=graph_data.edge_index.to(device),
        edge_attr=graph_data.edge_attr.to(device),
        y=graph_data.y.to(device)
    )
    
    # Move masks to device
    for key in ['train_mask', 'val_mask', 'test_mask']:
        split_dict[key] = split_dict[key].to(device)
    
    return data, split_dict, device
