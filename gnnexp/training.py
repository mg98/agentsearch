import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import logging
import os
from tqdm import tqdm
from gnnexp.models import EdgeRegressionModel
from data_utils import set_seed


def compute_metrics(predictions, targets):
    """Compute regression metrics"""
    with torch.no_grad():
        mse = nn.MSELoss()(predictions, targets)
        mae = nn.L1Loss()(predictions, targets)
        
        # Pearson correlation coefficient
        pred_mean = predictions.mean()
        target_mean = targets.mean()
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
        correlation = numerator / (denominator + 1e-8)
        
        # R2 score
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'rmse': torch.sqrt(mse).item(),
            'correlation': correlation.item(),
            'r2': r2.item()
        }


def train_epoch(model, data, split_dict, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    
    # Forward pass
    # we only have access to the edges in the current split
    train_predictions = model(data.x, data.edge_index[:, split_dict["train_mask"]], data.edge_attr[split_dict["train_mask"]])
    # predictions = model(data.x, data.edge_index, data.edge_attr)
    
    # Compute loss only on training edges
    # train_predictions = train_predictions[split_dict['train_mask']]
    train_targets = data.y[split_dict['train_mask']]

    loss = criterion(train_predictions, train_targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Compute metrics
    metrics = compute_metrics(train_predictions, train_targets)
    metrics['loss'] = loss.item()
    
    return metrics


def evaluate(model, data, split_dict, criterion, device, split='val'):
    """Evaluate model on validation or test set"""
    model.eval()
    
    with torch.no_grad():
        # a split can use only its and previous splits' edges for inference
        if split =='val':
            inclusive_mask = split_dict['train_mask'] + split_dict[f'val_mask']
            # predictions are of shape [inclusive_mask.sum()]
            # so we need to have a mask of this shape that tracks which ones are from split_dict['val_mask]
            output_mask = split_dict['val_mask'][inclusive_mask]
        elif split == 'test':
            inclusive_mask = torch.ones_like(data.y).bool()
            output_mask = split_dict['test_mask']
        else:
            raise ValueError(f"Unknown split: {split}")
        
    
        predictions = model(data.x, data.edge_index[:, inclusive_mask], data.edge_attr[inclusive_mask])
        
        split_predictions = predictions[output_mask]
        split_targets = data.y[split_dict[f'{split}_mask']]
        
        loss = criterion(split_predictions, split_targets)
        
        # Compute metrics
        metrics = compute_metrics(split_predictions, split_targets)
        metrics['loss'] = loss.item()
        
        return metrics


def train_model(model, data, split_dict, config, device):
    """Complete training loop"""
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize wandb
    wandb.init(
        project=config['experiment']['project_name'],
        name=config['experiment']['experiment_name'],
        config=config
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    logging.info("Starting training...")
    
    for epoch in tqdm(range(config['training']['epochs']), desc="Training"):
        # Train
        train_metrics = train_epoch(model, data, split_dict, optimizer, criterion, device)
        
        # Validate
        val_metrics = evaluate(model, data, split_dict, criterion, device, split='val')
        
        # Log metrics
        if epoch % config['logging']['log_every'] == 0:
            logging.info(f"Epoch {epoch:3d} | "
                        f"Train Loss: {train_metrics['loss']:.4f} | "
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val RMSE: {val_metrics['rmse']:.4f} | "
                        f"Val Corr: {val_metrics['correlation']:.4f}")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            **{f'train/{k}': v for k, v in train_metrics.items()},
            **{f'val/{k}': v for k, v in val_metrics.items()}
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
        
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    test_metrics = evaluate(model, data, split_dict, criterion, device, split='test')
    
    logging.info("Training completed!")
    logging.info(f"Final Test Metrics:")
    logging.info(f"  Test Loss: {test_metrics['loss']:.4f}")
    logging.info(f"  Test MAE: {test_metrics['mae']:.4f}")
    logging.info(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    logging.info(f"  Test Correlation: {test_metrics['correlation']:.4f}")
    
    # Log final test metrics to wandb
    wandb.log({
        **{f'final_test/{k}': v for k, v in test_metrics.items()}
    })
        
    wandb.finish()
    
    return test_metrics


def setup_model_and_training(graph_data, config):
    """Setup model and prepare for training"""
    
    # Set seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config['logging']['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize model
    model = EdgeRegressionModel(config)
    
    # Build model with data dimensions
    node_input_dim = graph_data.x.size(1)
    edge_input_dim = graph_data.edge_attr.size(1)
    model.build_model(node_input_dim, edge_input_dim)
    
    logging.info(f"Model initialized:")
    logging.info(f"  Node input dim: {node_input_dim}")
    logging.info(f"  Edge input dim: {edge_input_dim}")
    logging.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model
