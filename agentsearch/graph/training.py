import torch
import random
from copy import deepcopy
from agentsearch.dataset.questions import Question
from agentsearch.graph.types import TrustGNN, GraphData
import numpy as np

def train_model(model: TrustGNN, data: GraphData):
    epochs = 800
    lr = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_data, val_data = data.split()
    
    # Analyze data distribution
    print(f"Training on {train_data.edge_index.size(1)} edges, validating on {val_data.edge_index.size(1)} edges")
    
    # Check trust score distribution
    train_scores = train_data.edge_attributes[:, -1]
    val_scores = val_data.edge_attributes[:, -1]
    
    print(f"Train trust scores - Mean: {train_scores.mean():.3f}, Std: {train_scores.std():.3f}")
    print(f"Train trust scores - Min: {train_scores.min():.3f}, Max: {train_scores.max():.3f}")
    print(f"Val trust scores - Mean: {val_scores.mean():.3f}, Std: {val_scores.std():.3f}")
    print(f"Val trust scores - Min: {val_scores.min():.3f}, Max: {val_scores.max():.3f}")
    
    # Check for class imbalance
    train_positive = (train_scores > 0.5).float().mean()
    val_positive = (val_scores > 0.5).float().mean()
    print(f"Train positive ratio: {train_positive:.3f}")
    print(f"Val positive ratio: {val_positive:.3f}")
    
    print("--- Starting Training ---")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_pred_scores = model(train_data)
        train_loss = criterion(train_pred_scores, train_data.edge_attributes[:, -1])
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred_scores = model(val_data)
            val_loss = criterion(val_pred_scores, val_data.edge_attributes[:, -1])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
            
    print(f"--- Training Finished - Best Val Loss: {best_val_loss:.6f} ---")
    return best_val_loss

def evaluate_and_predict(model: TrustGNN, data: GraphData, title="Model Predictions"):
    _, data = data.split()
    print(f"\n--- {title} ---")
    model.eval()
    with torch.no_grad():
        pred_scores = model(data)
        # Apply sigmoid to convert logits to probabilities
        pred_scores = torch.sigmoid(pred_scores)

    print("Showing predictions for the first 10 edges:")
    print("-" * 45)
    print(f"{'Edge (S->T)':<15} | {'Actual Trust':<15} | {'Predicted Trust':<15}")
    print("-" * 45)

    for i in range(min(10, data.edge_index.size(1))):
        source = data.prediction_edge_index[0, i].item()
        target = data.prediction_edge_index[1, i].item()
        actual = data.edge_trust_score[i].item()
        predicted = pred_scores[i].item()
        print(f"Edge {source:>3} -> {target:<3} | {actual:<15.4f} | {predicted:<15.4f}")
    print("-" * 45)

def predict_top_targets(model: TrustGNN, data: GraphData, source_idx: int, question: Question, top_k: int = 10) -> list[tuple[int, float]]:
    """
    Simulate new edges from source_idx to all other nodes and predict which targets
    will yield the highest trust scores for the given question.
    
    Args:
        model: Trained TrustGNN model
        data: Graph data containing node embeddings
        source_idx: Source node index to create edges from
        question: Question object containing the query context
        top_k: Number of top predictions to return
    
    Returns:
        List of tuples (target_idx, predicted_score) sorted by score descending
    """
    model.eval()
    
    # Ensure question embedding is loaded
    if not isinstance(question.embedding, np.ndarray):
        question.load_embedding()
    
    # Get number of nodes in the graph
    num_nodes = data.x.size(0)
    
    # Create candidate target nodes (all nodes except the source)
    candidate_targets = [i for i in range(num_nodes) if i != source_idx]
    
    if len(candidate_targets) == 0:
        return []
    
    # Create new edges from source to all candidate targets
    num_new_edges = len(candidate_targets)
    new_edge_index = torch.tensor([
        [source_idx] * num_new_edges,  # Source nodes
        candidate_targets  # Target nodes
    ], dtype=torch.long)
    
    # Create edge features for new edges
    # Use a neutral trust score (0.5) as we're just predicting
    neutral_trust_scores = torch.full((num_new_edges, 1), 0.5, dtype=torch.float32)
    
    # Repeat the question embedding for all new edges
    question_emb_tensor = torch.tensor(question.embedding, dtype=torch.float32).unsqueeze(0)
    new_edge_query_embeddings = question_emb_tensor.repeat(num_new_edges, 1)

    new_edge_attributes = torch.cat([new_edge_query_embeddings, neutral_trust_scores], dim=1)
    
    # Combine existing edges with new edges for prediction
    temp_data = deepcopy(data)
    temp_data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
    temp_data.edge_attributes = torch.cat([data.edge_attributes, new_edge_attributes], dim=0)

    # Also update prediction-specific attributes
    temp_data.prediction_edge_index = temp_data.edge_index
    
    # Make predictions
    with torch.no_grad():
        pred_scores = model(temp_data)
        # Apply sigmoid to convert logits to probabilities
        pred_scores = torch.sigmoid(pred_scores)
    
    # Extract predictions for the new edges only (last num_new_edges predictions)
    new_edge_predictions = pred_scores[-num_new_edges:]
    
    # Create list of (target_idx, predicted_score) tuples
    predictions = []
    for i, target_idx in enumerate(candidate_targets):
        predicted_score = new_edge_predictions[i].item()
        predictions.append((target_idx, predicted_score))
    
    # Sort by predicted score in descending order and return top_k
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions[:top_k]
