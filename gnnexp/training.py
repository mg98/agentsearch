from gnnexp.models import HeadlessEdgeRegressionModel, EdgePredictionHead
from gnnexp.data_utils import set_seed

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import wandb
from tqdm import tqdm

import logging
import os


def compute_metrics_regression(predictions, targets):
    """Compute regression metrics"""
    with torch.no_grad():
        mse = nn.MSELoss()(predictions, targets)
        mae = nn.L1Loss()(predictions, targets)
        
        
        # R2 score
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'rmse': torch.sqrt(mse).item(),
            'r2': r2.item()
        }


def compute_metrics_binary_classification(probs: torch.Tensor, targets: torch.Tensor):
    """Compute binary classification metrics

    predictions: probabilities (after sigmoid)
    targets: binary labels (0 or 1)
    """

    with torch.no_grad():
        preds = (probs > 0.5).bool()
        targets = targets.bool()

        tp = (preds & targets).sum()
        fp = (preds & ~targets).sum()
        tn = (~preds & ~targets).sum()
        fn = (~preds & targets).sum()

        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(), 
            'tp': tp.item(),
            'fp': fp.item(),
            'tn': tn.item(),
            'fn': fn.item()
        }

def train_epoch(model, head, data, split_dict, optimizer, criterion, batchsize):
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
    all_preds = torch.zeros_like(data.y[split_dict['train_mask']]).long()  # to store predictions for the whole epoch
    all_tgts = data.y[split_dict['train_mask']][permutation].long()  # corresponding targets
    epoch_loss = 0.0

    for i in range(0, len(shuffled_indices), batchsize):
        # this removes the edges in the current batch from the training mask
        batch_indices = shuffled_indices[i:i + batchsize]
        batch_mask = torch.zeros_like(split_dict['train_mask'])
        batch_mask[batch_indices] = True

        inclusive_mask = split_dict['train_mask'] ^ batch_mask  # all edges except the current batch

        assert (inclusive_mask & batch_mask).sum() == 0, "The inclusive mask and batch mask should not overlap."

        # Forward pass with all train_edges (including labels!) except for the current batch
        # Ensure consistent dtype (avoid Double vs Float mismatch)
        model_dtype = next(model.parameters()).dtype
        edge_attributes_with_label = torch.cat((
            data.edge_attr[inclusive_mask].to(model_dtype),        # take all edge attributes except the current batch
            data.y[inclusive_mask].unsqueeze(1).to(model_dtype)),  # and add the trust score label
            dim=1
        )

        node_embeddings, _ = model(
            data.x.to(model_dtype),
            data.edge_index[:, inclusive_mask],  # take all edges except the current batch
            edge_attributes_with_label
        )

        # Add the head
        train_logits = head(node_embeddings, data.edge_attr[batch_mask].to(model_dtype), data.edge_index[:, batch_mask])

        # Get the targets for the current batch
        train_targets = data.y[batch_mask]

        loss = criterion(train_logits, train_targets)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Compute probabilities and predictions
        probs = torch.sigmoid(train_logits)

        # Store predictions for logging
        all_preds[i:i + batchsize] = probs
        epoch_loss += loss.item()


    # after the epoch is done, we step the optimizer
    optimizer.step()
    
    # Compute metrics on epoch
    metrics = compute_metrics_binary_classification(all_preds, all_tgts)
    metrics['loss'] = epoch_loss
    metrics['LR'] = optimizer.param_groups[0]['lr']

    return metrics


def evaluate(model, head, data, split_dict, criterion, split='val'):
    """Evaluate model on validation or test set"""
    model.eval()
    
    with torch.no_grad():
        # validation and test use only the train edges
        inclusive_mask = split_dict['train_mask'] 

        if split =='val':
            target_mask = split_dict['val_mask'] # the edges that we want to predict
        elif split == 'test':
            target_mask = split_dict['test_mask'] # the edges that we want to predict
        else:
            raise ValueError(f"Unknown split: {split}")

        assert (inclusive_mask & target_mask).sum() == 0, "The inclusive mask and target mask should not overlap."

        node_embeddings, _ = model(
            data.x, 
            data.edge_index[:, inclusive_mask], 
            torch.cat(
                (data.edge_attr[inclusive_mask],
                data.y[inclusive_mask].unsqueeze(1)),  # add the trust score label from the train
                dim=1
            )
        )

        split_logits = head(node_embeddings, data.edge_attr[target_mask], data.edge_index[:, target_mask])
        split_targets = data.y[target_mask]

        loss = criterion(split_logits, split_targets)

        # Compute probabilities and predictions
        split_probs = torch.sigmoid(split_logits)

        # Compute metrics
        metrics = compute_metrics_binary_classification(split_probs, split_targets.long())
        metrics['loss'] = loss.item()

        return metrics


def train_model(model, head, data, split_dict, config):
    """Complete training loop"""
    
    # Setup training
    # criterion = nn.MSELoss() 
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    # add head parameters to optimizer
    optimizer.add_param_group({'params': head.parameters()})
    
    # TODO Scheduler

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
        train_metrics = train_epoch(model, head, data, split_dict, optimizer, criterion,  
                                    config['training']['batchsize'])
        
        # Validate
        val_metrics = evaluate(model, head, data, split_dict, criterion, split='val')

        # Log metrics
        if epoch % config['logging']['log_every'] == 0:
            logging.info(f"Epoch {epoch:3d} | "
                        f"Train Loss: {train_metrics['loss']:.4f} | "
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Train acc: {train_metrics['accuracy']:.4f} | "
                        f"Val acc: {val_metrics['accuracy']:.4f} | ")
        
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
    test_metrics = evaluate(model, head, data, split_dict, criterion, split='test')
    
    logging.info("Training completed!")
    logging.info(f"Final Test Metrics:")
    for k, v in test_metrics.items():
        logging.info(f"  {k}: {v:.4f}")
    
    # Log final test metrics to wandb
    wandb.log({
        **{f'final_test/{k}': v for k, v in test_metrics.items()}
    })
        
    wandb.finish()
    
    return test_metrics


def setup_model_and_training(graph_data, config):
    """Setup model and prepare for training"""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config['logging']['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    node_input_dim = graph_data.x.size(1)
    edge_input_dim = graph_data.edge_attr.size(1)
    
    # Initialize model
    # the + 1 is for the trust score label
    model = HeadlessEdgeRegressionModel(
        node_input_dim, 
        edge_input_dim + 1,  # + 1 because we use label here
        config
    )

    # TODO should target edge also have an embedder?
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
    # print by component
    logging.info(f"     Node embedder parameters: {sum(p.numel() for p in model.node_embedder.parameters()):,}")
    logging.info(f"     Edge embedder parameters: {sum(p.numel() for p in model.edge_embedder.parameters()):,}")
    logging.info(f"     Message passing parameters: {sum(p.numel() for p in model.message_passing.parameters()):,}")
    logging.info(f"  Head parameters: {sum(p.numel() for p in head.parameters()):,}")
    
    return model, head
