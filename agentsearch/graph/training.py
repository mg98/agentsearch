import torch
from agentsearch.dataset.questions import Question
from agentsearch.graph.types import TrustGNN, GraphData

def train_model(model: TrustGNN, data: GraphData, epochs: int = 100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    print("--- Starting Training ---")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_scores = model(data)
        loss = criterion(pred_scores, data.edge_trust_score)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.6f}')
    print("--- Training Finished ---")

def evaluate_and_predict(model: TrustGNN, data: GraphData, title="Model Predictions"):
    print(f"\n--- {title} ---")
    model.eval()
    with torch.no_grad():
        pred_scores = model(data)

    print("Showing predictions for the first 10 edges:")
    print("-" * 45)
    print(f"{'Edge (S->T)':<15} | {'Actual Trust':<15} | {'Predicted Trust':<15}")
    print("-" * 45)

    for i in range(20):
        source = data.edge_index[0, i].item()
        target = data.edge_index[1, i].item()
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
    if question.embedding is None:
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
    
    # Combine existing edges with new edges for prediction
    combined_edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
    combined_edge_trust_scores = torch.cat([data.edge_trust_score, neutral_trust_scores], dim=0)
    combined_edge_query_embeddings = torch.cat([data.edge_query_embedding, new_edge_query_embeddings], dim=0)
    
    # Create temporary GraphData for prediction
    temp_data = GraphData(
        x=data.x,
        edge_index=combined_edge_index,
        edge_trust_score=combined_edge_trust_scores,
        edge_query_embedding=combined_edge_query_embeddings
    )
    
    # Make predictions
    with torch.no_grad():
        pred_scores = model(temp_data)
    
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
