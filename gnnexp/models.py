import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class NodeEmbedder(nn.Module):
    """Node feature embedder using MLP"""
    
    def __init__(self, input_dim, embedding_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embedding_dim * 2
            
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.layers(x)


class EdgeEmbedder(nn.Module):
    """Edge feature embedder using MLP"""
    
    def __init__(self, input_dim, embedding_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embedding_dim * 2
            
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, edge_attr):
        return self.layers(edge_attr)


class MessagePassingGCN(nn.Module):
    """Graph Convolutional Network for message passing"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EdgePredictionHead(nn.Module):
    """MLP head for edge prediction"""
    
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim=1, dropout=0.1):
        super().__init__()
        
        # Combine source node, target node, and edge features
        input_dim = 2 * node_dim + edge_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, node_embeddings, edge_embeddings, edge_index):
        # Get source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]  # Source nodes
        tgt_embeddings = node_embeddings[edge_index[1]]  # Target nodes
        
        # Concatenate source, target, and edge features
        edge_features = torch.cat([src_embeddings, tgt_embeddings, edge_embeddings], dim=1)
        
        # TODO squeeze the output to remove the last dimension

        return self.layers(edge_features)

class HeadlessEdgeRegressionModel(nn.Module):
    """Model for edge regression without Head, returning instead the learned node and edge embeddings"""
    
    def __init__(self, node_input_dim, edge_input_dim, config):
        super().__init__()
        
        self.config = config
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim

        # Initialize components
        self.node_embedder = NodeEmbedder(
            input_dim=node_input_dim,
            embedding_dim=self.config['model']['node_embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            dropout=self.config['model']['dropout']
        )
        
        self.edge_embedder = EdgeEmbedder(
            input_dim=edge_input_dim,
            embedding_dim=self.config['model']['edge_embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            dropout=self.config['model']['dropout']
        )
        
        self.message_passing = MessagePassingGCN(
            input_dim=self.config['model']['node_embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        )

    def forward(self, x, edge_index, edge_attr):
        # Embed nodes and edges
        node_embeddings = self.node_embedder(x)
        edge_embeddings = self.edge_embedder(edge_attr)
        
        # Message passing
        # TODO this needs to also include edge embeddings
        node_embeddings = self.message_passing(node_embeddings, edge_index)

        # Return node and edge embeddings
        return node_embeddings, edge_embeddings


class EdgeRegressionModel(nn.Module):
    """Complete model for edge regression"""
    
    def __init__(self, config):
        super().__init__()
        
        # Extract dimensions from data (will be set during initialization)
        self.node_input_dim = None
        self.edge_input_dim = None
        
        # Model components
        self.node_embedder = None
        self.edge_embedder = None
        self.message_passing = None
        self.prediction_head = None
        
        self.config = config
        
    def build_model(self, node_input_dim, edge_input_dim):
        """Build model components with known input dimensions"""
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        
        # Initialize components
        self.node_embedder = NodeEmbedder(
            input_dim=node_input_dim,
            embedding_dim=self.config['model']['node_embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            dropout=self.config['model']['dropout']
        )
        
        self.edge_embedder = EdgeEmbedder(
            input_dim=edge_input_dim,
            embedding_dim=self.config['model']['edge_embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            dropout=self.config['model']['dropout']
        )
        
        self.message_passing = MessagePassingGCN(
            input_dim=self.config['model']['node_embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        )
        
        self.prediction_head = EdgePredictionHead(
            node_dim=self.config['model']['hidden_dim'],
            edge_dim=self.config['model']['edge_embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            output_dim=1,
            dropout=self.config['model']['dropout']
        )
        
    def forward(self, x, edge_index, edge_attr):
        # Embed nodes and edges
        node_embeddings = self.node_embedder(x)
        edge_embeddings = self.edge_embedder(edge_attr)
        
        # Message passing
        node_embeddings = self.message_passing(node_embeddings, edge_index)
        
        # Edge prediction
        edge_predictions = self.prediction_head(node_embeddings, edge_embeddings, edge_index)
        
        return edge_predictions.squeeze(-1)  # Remove last dimension for regression
