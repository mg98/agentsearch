import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from agentsearch.utils.globals import EMBEDDING_DIM

class GraphData(Data):
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


class TrustGNNLayer(MessagePassing):
    def __init__(self, node_in_channels, node_out_channels, edge_feat_channels, dropout_rate):
        super().__init__(aggr='add') # 'add' or 'mean' aggregation
        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.node_lin = torch.nn.Linear(node_in_channels, node_out_channels)
        # MLP to process edge features before message passing
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_feat_channels, edge_feat_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),  # Add dropout in edge MLP
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
    def __init__(self, hidden_channels, node_out_channels, dropout_rate=0.2):
        super().__init__()
        node_in_channels = EMBEDDING_DIM
        # Concatenate trust score (1) and query embedding (query_embedding_dim)
        edge_feat_channels = 1 + EMBEDDING_DIM

        self.conv1 = TrustGNNLayer(node_in_channels, hidden_channels, edge_feat_channels, dropout_rate)
        # The second layer takes the output of the first layer as input
        self.conv2 = TrustGNNLayer(hidden_channels, node_out_channels, edge_feat_channels, dropout_rate)
        
        # Add dropout layers
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Predictor for trust score
        # Takes concatenated (source_node_embedding, target_node_embedding, query_embedding)
        # Input dimension: node_out_channels * 2 (for source and target) + query_embedding_dim
        predictor_in_dim = node_out_channels * 2 + EMBEDDING_DIM
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(predictor_in_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),  # Dropout after first layer
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid() # To output a score between 0 and 1
        )
        
        # Initialize the final layer bias to prevent sigmoid saturation
        with torch.no_grad():
            self.predictor[-2].bias.fill_(0.5)

    def forward(self, data):
        x, edge_index, edge_trust_score, edge_query_embedding = data.x, data.edge_index, data.edge_trust_score, data.edge_query_embedding

        # Combine edge trust score and query embedding for message passing
        edge_attr_for_prop = torch.cat([edge_trust_score, edge_query_embedding], dim=1)

        # 1. Propagate messages to get final node embeddings
        x = self.conv1(x, edge_index, edge_attr_for_prop)
        x = self.dropout(x)  # Dropout after first conv layer
        x = self.conv2(x, edge_index, edge_attr_for_prop)
        x = self.dropout(x)  # Dropout after second conv layer

        # 2. Predict trust scores for all edges
        row, col = edge_index
        source_node_emb = x[row]
        target_node_emb = x[col]

        # Concatenate for the predictor
        # Input shape: [num_edges, 2 * node_out_channels + query_embedding_dim]
        predictor_input = torch.cat([source_node_emb, target_node_emb, edge_query_embedding], dim=-1)

        trust_predictions = self.predictor(predictor_input)

        return trust_predictions
