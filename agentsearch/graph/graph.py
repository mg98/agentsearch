import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.dataset import agents
from agentsearch.agent import eval

# 1. Data Representation (User-provided class)
class CustomGraphData(Data):
    """
    Custom Data object to hold node features (embeddings)
    and edge features (trust_score, query_embedding).
    """
    def __init__(self, x=None, edge_index=None, edge_trust_score=None, edge_query_embedding=None):
        super().__init__()
        self.x = x  # Node embeddings (shape: num_nodes, node_embedding_dim)
        self.edge_index = edge_index  # Adjacency list (shape: 2, num_edges)
        self.edge_trust_score = edge_trust_score # Trust score for each edge (shape: num_edges, 1)
        self.edge_query_embedding = edge_query_embedding # Query embedding for each edge (shape: num_edges, query_embedding_dim)

# 2. Define a Custom GNN Layer (User-provided class)
class TrustGNNLayer(MessagePassing):
    def __init__(self, node_in_channels, node_out_channels, edge_feat_channels):
        super().__init__(aggr='add') # 'add' or 'mean' aggregation
        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.node_lin = torch.nn.Linear(node_in_channels, node_out_channels)
        # MLP to process edge features before message passing
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_feat_channels, edge_feat_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_feat_channels * 2, node_in_channels) # Output matches node_in_channels for addition in message
        )

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_feat_channels]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j has shape [E, in_channels] (neighbor node features)
        # edge_attr has shape [E, edge_feat_channels] (edge features)
        edge_transformed = self.edge_mlp(edge_attr)
        # Combine neighbor node features with transformed edge features
        return x_j + edge_transformed

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels] (aggregated messages)
        # Apply a linear transformation and activation to the aggregated messages
        return F.relu(self.node_lin(aggr_out))

# 3. Define the Full TrustGNN Model (MODIFIED FOR TRAINING)
class TrustGNN(torch.nn.Module):
    def __init__(self, node_in_channels, hidden_channels, node_out_channels, query_embedding_dim):
        super().__init__()
        # Concatenate trust score (1) and query embedding (query_embedding_dim)
        edge_feat_channels = 1 + query_embedding_dim

        self.conv1 = TrustGNNLayer(node_in_channels, hidden_channels, edge_feat_channels)
        # The second layer takes the output of the first layer as input
        self.conv2 = TrustGNNLayer(hidden_channels, node_out_channels, edge_feat_channels)

        # Predictor for trust score
        # Takes concatenated (source_node_embedding, target_node_embedding, query_embedding)
        # Input dimension: node_out_channels * 2 (for source and target) + query_embedding_dim
        predictor_in_dim = node_out_channels * 2 + query_embedding_dim
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(predictor_in_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid() # To output a score between 0 and 1
        )

    def forward(self, data):
        x, edge_index, edge_trust_score, edge_query_embedding = data.x, data.edge_index, data.edge_trust_score, data.edge_query_embedding

        # Combine edge trust score and query embedding for message passing
        edge_attr_for_prop = torch.cat([edge_trust_score, edge_query_embedding], dim=1)

        # 1. Propagate messages to get final node embeddings
        x = self.conv1(x, edge_index, edge_attr_for_prop)
        x = self.conv2(x, edge_index, edge_attr_for_prop)

        # 2. Predict trust scores for all edges
        row, col = edge_index
        source_node_emb = x[row]
        target_node_emb = x[col]

        # Concatenate for the predictor
        # Input shape: [num_edges, 2 * node_out_channels + query_embedding_dim]
        predictor_input = torch.cat([source_node_emb, target_node_emb, edge_query_embedding], dim=-1)

        trust_predictions = self.predictor(predictor_input)

        return trust_predictions

# 4. Training Function
def train_model(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    print("--- Starting Training ---")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass: model predicts scores for all edges
        pred_scores = model(data)

        # Compute loss against ground truth scores
        loss = criterion(pred_scores, data.edge_trust_score)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.6f}')
    print("--- Training Finished ---")

# 5. Prediction and Evaluation Function
def evaluate_and_predict(model, data, title="Model Predictions"):
    print(f"\n--- {title} ---")
    model.eval()
    with torch.no_grad():
        pred_scores = model(data)

    print("Showing predictions for the first 10 edges:")
    print("-" * 45)
    print(f"{'Edge (S->T)':<15} | {'Actual Trust':<15} | {'Predicted Trust':<15}")
    print("-" * 45)

    for i in range(10):
        source = data.edge_index[0, i].item()
        target = data.edge_index[1, i].item()
        actual = data.edge_trust_score[i].item()
        predicted = pred_scores[i].item()
        print(f"Edge {source:>3} -> {target:<3} | {actual:<15.4f} | {predicted:<15.4f}")
    print("-" * 45)


# 6. Graph Visualization Function
def visualize_graph(data, title="Graph Visualization"):
    """Visualizes the graph with edge colors based on trust score."""
    g = nx.DiGraph() # Use DiGraph for directed edges

    # Add nodes
    g.add_nodes_from(range(data.x.shape[0]))

    # Add edges and trust scores as attributes
    edges = data.edge_index.t().cpu().numpy()
    trust_scores = data.edge_trust_score.cpu().numpy().flatten()
    g.add_edges_from(edges)

    # Use trust scores for edge colors, ensuring it's a flattened float array and handling potential NaNs
    edge_colors = trust_scores.astype(float).flatten()
    # Replace any non-finite values (like NaNs) with a default value (e.g., 0.0)
    edge_colors[np.isnan(edge_colors)] = 0.0

    # Create a ScalarMappable to map trust scores to colors
    cmap = cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([]) # Or pass the actual data: sm.set_array(edge_colors)

    # Map the edge_colors data to RGBA values using the ScalarMappable
    rgba_colors = sm.to_rgba(edge_colors)

    # Explicitly create figure and axes
    fig, ax = plt.subplots(figsize=(12, 12))

    pos = nx.spring_layout(g, seed=42) # for reproducible layout
    nx.draw_networkx_nodes(g, pos, node_color='lightblue', node_size=200, ax=ax)
    edges = nx.draw_networkx_edges(g, pos, edge_color=rgba_colors, # Use the mapped RGBA colors
                                   arrows=True, arrowstyle='->', arrowsize=10, width=1.5, ax=ax)

    # Add a colorbar, explicitly linked to the axes
    cbar = plt.colorbar(sm, shrink=0.8, ax=ax) # Use the ScalarMappable for the colorbar and link to axes
    cbar.set_label('Edge Trust Score', rotation=270, labelpad=15)

    ax.set_title(title, fontsize=16)
    plt.show()


# --- Main Execution ---
if __name__ == '__main__':

    Edge = tuple[int, int, np.ndarray, int] # (source_node_idx, target_node_idx, query_embedding, trust_score)
    edges: list[Edge] = []

    # Get all agents and create ID to index mapping
    all_agents = Agent.all()
    agent_id_to_index = {agent.id: idx for idx, agent in enumerate(all_agents)}
    print(agent_id_to_index.keys())
    
    i = 0
    for question in Question.all():
        i += 1
        if i > 50:
            break
        print(question.question)
        print("-"*100)

        # evaluate top-5 agents
        for match in agents.match_by_qid(question.id, 5, blacklist=[question.agent_id]):
            print(match.agent.name)
            print(match.agent.scholar_url)
            print(match.similarity_score)

            answer = match.agent.ask(question.question)
            print("Answer:", answer)

            grade, reason = eval.grade_answer(question.question, answer)
            print("Grade:", grade)
            print("Reason:", reason)
            print("-"*100)

            # Use graph node indices instead of agent CSV IDs
            source_idx = agent_id_to_index[question.agent_id]
            target_idx = agent_id_to_index[match.agent.id]
            assert question.embedding is not None
            assert len(question.embedding) == 1024
            edges.append((source_idx, target_idx, question.embedding, grade))

    # Use the same agents list for embeddings
    node_embeddings = [agent.embedding for agent in all_agents]
    node_embeddings = np.array(node_embeddings)
    node_embeddings = torch.from_numpy(node_embeddings).float()
    num_nodes = len(node_embeddings)

    assert node_embeddings.shape[1] == 1024
    node_embedding_dim = 1024
    query_embedding_dim = 1024
    num_edges = len(edges)

    source_ids = [edge[0] for edge in edges]
    target_ids = [edge[1] for edge in edges]
    query_embeddings = [edge[2] for edge in edges]
    trust_scores = [edge[3] for edge in edges]
    
    edge_index = torch.tensor([source_ids, target_ids], dtype=torch.long)
    edge_trust_scores = torch.tensor(trust_scores, dtype=torch.float).unsqueeze(1)  # Shape: (num_edges, 1)
    
    edge_query_embeddings = torch.tensor(np.stack(query_embeddings), dtype=torch.float)

    data = CustomGraphData(
        x=node_embeddings,
        edge_index=edge_index,
        edge_trust_score=edge_trust_scores,
        edge_query_embedding=edge_query_embeddings
    )

    print("Graph Data Summary:")
    print(data)
    print(f"Agent ID to Index mapping created for {len(agent_id_to_index)} agents")
    print(f"Node indices range: 0-{num_nodes-1}")
    print(f"Edge source indices: {set(source_ids)}")
    print(f"Edge target indices: {set(target_ids)}")

    # Instantiate the model
    model = TrustGNN(
        node_in_channels=node_embedding_dim,
        hidden_channels=128,
        node_out_channels=64,
        query_embedding_dim=query_embedding_dim
    )
    print("\nModel Architecture:")
    print(model)

    # 1. Visualize the initial graph
    visualize_graph(data, title=f"Graph with {num_nodes} Nodes and {data.num_edges} Edges")

    # 2. Show predictions with the untrained model
    evaluate_and_predict(model, data, title="Predictions Before Training")

    # 3. Train the model
    train_model(model, data, epochs=100)

    # 4. Show predictions with the trained model
    evaluate_and_predict(model, data, title="Predictions After Training")