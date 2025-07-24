import torch
import random
from copy import deepcopy
from agentsearch.dataset.questions import Question
from agentsearch.graph.types import TrustGNN, GraphData
import numpy as np
import torch.nn.functional as F

torch.manual_seed(42)

def train_model(model: TrustGNN, data: GraphData):
    epochs = 100
    lr = 0.001  # Higher learning rate for simpler model
    print(f"Learning rate: {lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Use CrossEntropy for classification task
    criterion = torch.nn.CrossEntropyLoss()

    train_data, val_data = data.split()
    
    # Convert trust scores to class labels
    def trust_to_class(trust_scores):
        """Convert continuous trust scores to class labels"""
        classes = torch.zeros_like(trust_scores, dtype=torch.long)
        classes[trust_scores == 0.0] = 0  # Low trust
        classes[trust_scores == 0.5] = 1  # Medium trust  
        classes[trust_scores == 1.0] = 2  # High trust
        return classes
    
    train_labels = trust_to_class(train_data.edge_attributes[:, -1])
    val_labels = trust_to_class(val_data.edge_attributes[:, -1])
    
    # Analyze data distribution
    print(f"Training on {train_data.edge_index.size(1)} edges, validating on {val_data.edge_index.size(1)} edges")
    
    # Check class distribution
    train_counts = torch.bincount(train_labels, minlength=3)
    val_counts = torch.bincount(val_labels, minlength=3)
    
    print(f"Train class distribution - Low: {train_counts[0]}, Medium: {train_counts[1]}, High: {train_counts[2]}")
    print(f"Val class distribution - Low: {val_counts[0]}, Medium: {val_counts[1]}, High: {val_counts[2]}")
    
    print("--- Starting Training ---")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Get logits for CrossEntropy loss
        train_logits = model.forward_logits(train_data)
        train_loss = criterion(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model.forward_logits(val_data)
            val_loss = criterion(val_logits, val_labels)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        if (epoch + 1) % 10 == 0:
            # Calculate accuracy
            train_pred = torch.argmax(train_logits, dim=1)
            val_pred = torch.argmax(val_logits, dim=1)
            train_acc = (train_pred == train_labels).float().mean()
            val_acc = (val_pred == val_labels).float().mean()
            
            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}')
            
    print(f"--- Training Finished - Best Val Loss: {best_val_loss:.6f} ---")
    return best_val_loss

def evaluate_and_predict(model: TrustGNN, data: GraphData, title="Model Predictions"):
    _, data = data.split()
    print(f"\n--- {title} ---")
    model.eval()
    with torch.no_grad():
        pred_scores = model(data)
        # Model already outputs probabilities in [0,1] range

    print("Showing predictions for the first 20 edges:")
    print("-" * 45)
    print(f"{'Edge (S->T)':<15} | {'Actual Trust':<15} | {'Predicted Trust':<15}")
    print("-" * 45)

    for i in range(min(20, data.edge_index.size(1))):
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
    
    # Update trust_scores if it exists
    if hasattr(data, 'trust_scores') and isinstance(data.trust_scores, torch.Tensor):
        new_trust_scores = torch.full((num_new_edges,), 0.5, dtype=torch.float)
        temp_data.trust_scores = torch.cat([data.trust_scores, new_trust_scores], dim=0)

    # Set up prediction-specific attributes that the model expects
    temp_data.prediction_edge_index = temp_data.edge_index
    temp_data.edge_trust_score = temp_data.edge_attributes[:, -1].unsqueeze(1)
    temp_data.edge_query_embedding = temp_data.edge_attributes[:, :-1]
    temp_data.prediction_source_ids = temp_data.edge_index[0]
    temp_data.prediction_target_ids = temp_data.edge_index[1]
    
    # Make predictions
    with torch.no_grad():
        pred_scores = model(temp_data)
        # Model already outputs probabilities in [0,1] range
    
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
    
    # Update trust_scores if it exists
    if hasattr(data, 'trust_scores') and isinstance(data.trust_scores, torch.Tensor):
        new_trust_scores = torch.full((num_new_edges,), 0.5, dtype=torch.float)
        temp_data.trust_scores = torch.cat([data.trust_scores, new_trust_scores], dim=0)

    # Set up prediction-specific attributes that the model expects
    temp_data.prediction_edge_index = temp_data.edge_index
    temp_data.edge_trust_score = temp_data.edge_attributes[:, -1].unsqueeze(1)
    temp_data.edge_query_embedding = temp_data.edge_attributes[:, :-1]
    temp_data.prediction_source_ids = temp_data.edge_index[0]
    temp_data.prediction_target_ids = temp_data.edge_index[1]
    
    # Make predictions
    with torch.no_grad():
        pred_scores = model(temp_data)
    
    # Extract predictions for the new edges only (last num_new_edges predictions)
    new_edge_predictions = pred_scores[-num_new_edges:]
    
    # Return as list of floats
    return [score.item() for score in new_edge_predictions]
