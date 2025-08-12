import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
import numpy as np
from agentsearch.graph.types import GraphData, TrustGNN
from agentsearch.dataset.questions import Question

torch.manual_seed(42)

def train_model(model: TrustGNN, data: GraphData):
    """Train the TrustGNN model with enhanced features and diversity regularization"""
    print(f"Learning rate: {0.01}")
    
    # Split data
    train_data, val_data = data.split(val_ratio=0.2)
    
    print(f"Training on {train_data.edge_index.size(1)} edges, validating on {val_data.edge_index.size(1)} edges")
    
    # Print statistics
    train_trust_stats = train_data.edge_trust_score.squeeze()
    val_trust_stats = val_data.edge_trust_score.squeeze()
    print(f"Train trust score stats - Min: {train_trust_stats.min():.3f}, Max: {train_trust_stats.max():.3f}, Mean: {train_trust_stats.mean():.3f}, Std: {train_trust_stats.std():.3f}")
    print(f"Val trust score stats - Min: {val_trust_stats.min():.3f}, Max: {val_trust_stats.max():.3f}, Mean: {val_trust_stats.mean():.3f}, Std: {val_trust_stats.std():.3f}")
    
    # Initialize optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)
    
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    max_patience = 10
    
    print("--- Starting Training ---")
    for epoch in range(1, 201):
        model.train()
        
        # Forward pass
        pred_scores = model(train_data)
        
        # Main MSE loss
        mse_loss = criterion(pred_scores, train_data.edge_trust_score.squeeze())
        
        # Add diversity regularization to prevent agent bias
        target_agents = train_data.prediction_target_ids
        unique_agents, agent_counts = torch.unique(target_agents, return_counts=True)
        
        # Compute agent-specific average predictions
        agent_avg_preds = torch.zeros(len(unique_agents))
        for i, agent_id in enumerate(unique_agents):
            agent_mask = target_agents == agent_id
            agent_avg_preds[i] = pred_scores[agent_mask].mean()
        
        # Diversity loss - penalize if agent predictions are too similar
        if len(agent_avg_preds) > 1:
            diversity_loss = -torch.var(agent_avg_preds) * 0.01  # Small weight
        else:
            diversity_loss = 0.0
        
        # Total loss
        total_loss = mse_loss + diversity_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred_scores = model(val_data)
            val_loss = criterion(val_pred_scores, val_data.edge_trust_score.squeeze())
            
            # Compute MAE for both train and val
            train_mae = torch.mean(torch.abs(pred_scores - train_data.edge_trust_score.squeeze())).item()
            val_mae = torch.mean(torch.abs(val_pred_scores - val_data.edge_trust_score.squeeze())).item()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Track best model
        if val_mae < best_val_mae:
            best_val_loss = val_loss.item()
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress every 3 epochs
        if epoch % 3 == 0:
            pred_min, pred_max = pred_scores.min().item(), pred_scores.max().item()
            pred_std = pred_scores.std().item()
            print(f"Epoch: {epoch:03d}, Train Loss: {total_loss.item():.6f}, Val Loss: {val_loss.item():.6f}, "
                  f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, "
                  f"Pred Range: [{pred_min:.3f}, {pred_max:.3f}], Pred Std: {pred_std:.3f}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}. Best val MAE: {best_val_mae:.4f}")
            break
    
    print(f"--- Training Finished - Best Val Loss: {best_val_loss:.6f}, Best Val MAE: {best_val_mae:.4f} ---")
    return best_val_loss


def evaluate_and_predict(model: TrustGNN, data: GraphData, title="Evaluation"):
    """
    Evaluate model performance and show sample predictions
    """
    model.eval()
    
    # Run evaluation on validation set
    _, val_data = data.split(val_ratio=0.2)
    
    with torch.no_grad():
        pred_scores = model(val_data)
        actual_scores = val_data.edge_trust_score.squeeze()
        
        # Calculate metrics
        mae = torch.mean(torch.abs(pred_scores - actual_scores)).item()
        mse = torch.mean((pred_scores - actual_scores) ** 2).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        print(f"\n--- {title} ---")
        print(f"Evaluation Metrics - MAE: {mae:.4f}, MSE: {mse:.6f}, RMSE: {rmse:.4f}")
        
        # Show predictions for first 20 edges
        print("Showing predictions for the first 20 edges:")
        print("-" * 60)
        print(f"{'Edge (S->T)':<15} | {'Actual (norm)':<15} | {'Predicted (norm)':<16} | {'Error':<9}")
        print("-" * 60)
        
        for i in range(min(20, len(pred_scores))):
            source_idx = val_data.prediction_source_ids[i].item()
            target_idx = val_data.prediction_target_ids[i].item()
            actual = actual_scores[i].item()
            predicted = pred_scores[i].item()
            error = abs(actual - predicted)
            
            print(f"Edge {source_idx:3d} -> {target_idx:<3d} | {actual:<15.4f} | {predicted:<16.4f} | {error:<9.4f}")
        
        print("-" * 60)


def predict_top_targets(model: TrustGNN, data: GraphData, source_idx: int, question: Question, top_k: int = 10) -> list[tuple[int, float]]:
    """
    Predict the top-k target agents for a given source and question.
    
    Args:
        model: Trained TrustGNN model
        data: Graph data containing node embeddings
        source_idx: Source node index
        question: Question object containing the query context
        top_k: Number of top predictions to return
    
    Returns:
        List of (target_index, predicted_score) tuples, sorted by score descending
    """
    model.eval()
    
    # Ensure question embedding is loaded
    if not isinstance(question.embedding, np.ndarray):
        question.load_embedding()
    
    # Create prediction data for all possible target agents (excluding source)
    all_target_indices = [i for i in range(len(data.agents)) if i != source_idx]
    
    # Predict trust scores for all possible targets
    predicted_scores = predict_trust_scores(model, data, source_idx, all_target_indices, question)
    
    # Create list of (target_index, score) tuples and sort by score
    target_score_pairs = list(zip(all_target_indices, predicted_scores))
    target_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return target_score_pairs[:top_k]


def predict_trust_scores(model: TrustGNN, data: GraphData, source_idx: int, target_indices: list[int], question: Question) -> list[float]:
    """
    Predict trust scores for specific source-target pairs given a question.
    
    Args:
        model: Trained TrustGNN model
        data: Graph data containing node embeddings
        source_idx: Source node index
        target_indices: List of target node indices to predict trust scores for
        question: Question object containing the query context
    
    Returns:
        List of predicted trust scores for each target
    """
    model.eval()
    
    # Ensure question embedding is loaded
    if not isinstance(question.embedding, np.ndarray):
        question.load_embedding()
    
    if len(target_indices) == 0:
        return []
    
    # Create new edges from source to specific targets
    num_new_edges = len(target_indices)
    new_edge_index = torch.tensor([
        [source_idx] * num_new_edges,  # Source nodes
        target_indices  # Target nodes
    ], dtype=torch.long)
    
    # Create edge features for new edges
    # Use the mean trust score from existing data instead of fixed 0.5
    mean_trust = data.trust_scores.mean().item() if data.trust_scores.size(0) > 0 else 0.3
    
    # Repeat the question embedding for all new edges (just embeddings, no trust scores)
    question_emb_tensor = torch.tensor(question.embedding, dtype=torch.float32).unsqueeze(0)
    new_edge_attributes = question_emb_tensor.repeat(num_new_edges, 1)
    
    # Combine existing edges with new edges for prediction
    temp_data = deepcopy(data)
    temp_data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
    temp_data.edge_attributes = torch.cat([data.edge_attributes, new_edge_attributes], dim=0)
    
    # Update trust_scores to match the new edges
    new_trust_scores = torch.full((num_new_edges,), mean_trust, dtype=torch.float)
    temp_data.trust_scores = torch.cat([data.trust_scores, new_trust_scores], dim=0)

    # Set up prediction-specific attributes that the model expects
    temp_data.prediction_edge_index = new_edge_index  # Only predict on NEW edges
    temp_data.edge_trust_score = new_trust_scores.unsqueeze(1)
    temp_data.edge_query_embedding = new_edge_attributes
    temp_data.prediction_source_ids = new_edge_index[0]
    temp_data.prediction_target_ids = new_edge_index[1]
    
    # Make predictions
    with torch.no_grad():
        pred_scores = model(temp_data)
    
    # Return as list of floats
    return [score.item() for score in pred_scores]