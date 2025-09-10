import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class NodeEmbedder(nn.Module):
    """Node feature embedder using MLP"""
    
    def __init__(self, input_dim, embedding_dim, dropout=0.1):
        super().__init__()
            
        self.layers = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.layers(x)


class EdgeEmbedder(nn.Module):
    """Edge feature embedder using MLP"""
    
    def __init__(self, input_dim, embedding_dim, dropout=0.1):
        super().__init__()
            
        self.layers = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
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


class EdgeUpdatingGAT(nn.Module):
    """GAT with Edge Updates"""

    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT Conv layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim_node, hidden_dim, dropout=dropout, edge_dim=input_dim_edge))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, dropout=dropout, edge_dim=hidden_dim))
        
        # Linear edge update layers
        self.edge_updates = nn.ModuleList()
        self.edge_updates.append(nn.Linear(2 * hidden_dim + input_dim_edge, hidden_dim))
        for _ in range(num_layers - 1):
            self.edge_updates.append(nn.Linear(3 * hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_layers):
            # Apply GAT Conv
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)

            # Update edge attributes
            edge_attr = self.edge_updates[i](torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=1))

        return x, edge_attr


class EdgePredictionHead(nn.Module):
    """MLP head for edge prediction"""
    
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim=1, dropout=0.1):
        super().__init__()
        

        # one layer to embed the target edge
        self.edge_embedder = nn.Linear(edge_dim, hidden_dim)

        # Combine source node, target node, and edge features
        input_dim = (2 * node_dim) + hidden_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, node_embeddings, edge_attrs, edge_index):
        # Get source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]  # Source nodes
        tgt_embeddings = node_embeddings[edge_index[1]]  # Target nodes

        edge_embeddings = self.edge_embedder(edge_attrs)

        # Concatenate Source, Target, and Edge Features
        edge_features = torch.cat([src_embeddings, tgt_embeddings, edge_embeddings], dim=1)
        
        return self.layers(edge_features).squeeze(-1)


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
            dropout=self.config['model']['dropout']
        )
        
        self.edge_embedder = EdgeEmbedder(
            input_dim=edge_input_dim,
            embedding_dim=self.config['model']['edge_embedding_dim'],
            dropout=self.config['model']['dropout']
        )
        
        self.message_passing = EdgeUpdatingGAT(
            input_dim_node=self.config['model']['node_embedding_dim'],
            input_dim_edge=self.config['model']['edge_embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        )

    def forward(self, x, edge_index, edge_attr):
        # Embed nodes and edges
        node_embeddings = self.node_embedder(x)
        edge_embeddings = self.edge_embedder(edge_attr)
        
        # Message passing
        node_embeddings, edge_embeddings = self.message_passing(node_embeddings, edge_index, edge_embeddings)

        # Return node and edge embeddings
        return node_embeddings, edge_embeddings
