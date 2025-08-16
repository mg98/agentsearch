import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import logging
import os
from tqdm import tqdm
from gnnexp.models import HeadlessEdgeRegressionModel, EdgePredictionHead
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


def train_epoch(model, head, data, split_dict, optimizer, criterion, device, batchsize=64):
    """Train for one epoch"""
    model.train()
    
    # we take batches from split_dict['train_indices']
    # the edges in the batch are removed from the graph
    # the labels of all the remaining edges are added to the edge attributes
    # the a forward pass of MPNN is done over this graph, giving node and edge embeddings
    # then we compute predictions for the batch edges using these embeddings and a head model

    # take random batches from split_dict['train_indices'] 

    # shuffle the indices for randomness
    permutation = torch.randperm(split_dict['train_indices'].shape[0])
    shuffled_indices = split_dict['train_indices'][permutation]

    # for logging epoch stats
    all_preds = torch.zeros_like(data.y[split_dict['train_mask']])  # to store predictions for the whole epoch
    all_tgts = data.y[split_dict['train_mask']][permutation]
    epoch_loss = 0.0

    for i in range(0, len(shuffled_indices), batchsize):
        # this removes the edges in the current batch from the training mask
        batch_indices = shuffled_indices[i:i + batchsize]
        batch_mask = torch.zeros_like(split_dict['train_mask'])
        batch_mask[batch_indices] = True

        inclusive_mask = split_dict['train_mask'] ^ batch_mask  # all edges except the current batch

        assert (inclusive_mask & batch_mask).sum() == 0, "The inclusive mask and batch mask should not overlap."

        # TODO we should probably put the head in a model
        # Forward pass with all train_edges (including labels!) except for the current batch
        node_embeddings, _ = model(
            data.x,
            data.edge_index[:, inclusive_mask],  # take all edges except the current batch
            torch.cat((
                data.edge_attr[inclusive_mask],      # take all edge attributes except the current batch
                data.y[inclusive_mask].unsqueeze(1)),  # and add the trust score label
                dim=1
            )
        )

        # Add the head
        # TODO do we need an embedding model for the edges?
        train_predictions = head(node_embeddings, data.edge_attr[batch_mask], data.edge_index[:, batch_mask])
        train_predictions = train_predictions.squeeze(1)

        # Get the targets for the current batch
        train_targets = data.y[batch_mask]

        loss = criterion(train_predictions, train_targets)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Store predictions for logging
        all_preds[i:i + batchsize] = train_predictions
        epoch_loss += loss.item()

    # after the epoch is done, we step the optimizer
    optimizer.step()
    
    # Compute metrics on the last batch
    metrics = compute_metrics(all_preds, all_tgts)
    metrics['loss'] = epoch_loss / len(shuffled_indices)
    metrics['LR'] = optimizer.param_groups[0]['lr']


    return metrics


def evaluate(model, head, data, split_dict, criterion, device, split='val'):
    """Evaluate model on validation or test set"""
    model.eval()
    
    with torch.no_grad():
        # a split can use only its and previous splits' edges for inference
        if split =='val':
            inclusive_mask = split_dict['train_mask'] # validation uses train edges and their labels for message passing
            target_mask = split_dict['val_mask'] # the edges that we want to predict
        elif split == 'test':
            inclusive_mask = split_dict['train_mask'] + split_dict[f'val_mask'] # test uses train and val edges
            target_mask = split_dict['test_mask'] # the edges that we want to predict
        else:
            raise ValueError(f"Unknown split: {split}")

        assert (inclusive_mask & target_mask).sum() == 0, "The inclusive mask and target mask should not overlap."

        node_embeddings, _ = model(
            data.x, 
            data.edge_index[:, inclusive_mask], 
            torch.cat(
                (data.edge_attr[inclusive_mask],
                data.y[inclusive_mask].unsqueeze(1)),  # add the trust score label
                dim=1
            )
        )

        split_predictions = head(node_embeddings, data.edge_attr[target_mask], data.edge_index[:, target_mask])
        split_predictions = split_predictions.squeeze(1)
        split_targets = data.y[target_mask]

        loss = criterion(split_predictions, split_targets)
        
        # Compute metrics
        metrics = compute_metrics(split_predictions, split_targets)
        metrics['loss'] = loss.item()


        return metrics


def train_model(model, head, data, split_dict, config, device):
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
        train_metrics = train_epoch(model, head, data, split_dict, optimizer, criterion, device, 
                                    batchsize=config['training']['batchsize'])
        
        # Validate
        val_metrics = evaluate(model, head, data, split_dict, criterion, device, split='val')
        
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
    test_metrics = evaluate(model, head, data, split_dict, criterion, device, split='test')
    
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
    node_input_dim = graph_data.x.size(1)
    edge_input_dim = graph_data.edge_attr.size(1)
    
    # Initialize model
    # the + 1 is for the trust score label
    model = HeadlessEdgeRegressionModel(node_input_dim, edge_input_dim + 1, config)
    head = EdgePredictionHead(
        node_dim=config['model']['hidden_dim'],
        edge_dim=edge_input_dim,
        hidden_dim=config['model']['hidden_dim'],
        output_dim=1,
        dropout=config['model']['dropout']
    )
    
    logging.info(f"Model initialized:")
    logging.info(f"  Node input dim: {node_input_dim}")
    logging.info(f"  Edge input dim: {edge_input_dim}")
    logging.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, head
