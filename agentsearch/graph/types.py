import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from agentsearch.utils.globals import EMBEDDING_DIM
from agentsearch.dataset.agents import Agent
import random
import numpy as np
from typing import List, Tuple
from collections import defaultdict
import networkx as nx

class GraphData(Data):
    def __init__(self, agents: list[Agent]):
        super().__init__()
        self.agents = agents
        self.agent_id_to_index = {agent.id: i for i, agent in enumerate(agents)}
        
        # Initialize edge structures
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.edge_attributes = torch.empty((0, EMBEDDING_DIM + 1))
        
        # Trust scores for each edge (will be populated as edges are added)
        self.trust_scores = torch.empty(0, dtype=torch.float)
        
        # Build rich node features
        self.x = self._build_node_features()
        
    def _build_node_features(self) -> torch.Tensor:
        """
        Build node feature matrix using agent embeddings instead of one-hot encoding.
        This creates meaningful, low-dimensional representations.
        """
        num_nodes = len(self.agents)
        
        # Use agent embeddings as primary node features
        node_features = []
        for agent in self.agents:
            if agent.embedding is None:
                agent.load_embedding()
            node_features.append(agent.embedding)
        
        # Convert to tensor
        embedding_features = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        # Add simple degree features (will be updated after edges are added)
        degree_features = torch.zeros(num_nodes, 4)  # in_degree, out_degree, avg_in_trust, avg_out_trust
        
        # Combine features (much smaller: EMBEDDING_DIM + 4 instead of 74)
        node_features = torch.cat([embedding_features, degree_features], dim=1)
        
        return node_features
    

    
    def add_edge(self, source_agent: Agent, target_agent: Agent, question_embedding: np.ndarray, grade: float):
        """Add an edge with trust score and question embedding"""
        source_idx = self.agent_id_to_index[source_agent.id]
        target_idx = self.agent_id_to_index[target_agent.id]

        new_edge = torch.tensor([[source_idx], [target_idx]], dtype=torch.long)
        self.edge_index = torch.cat([self.edge_index, new_edge], dim=1)

        edge_attr = torch.cat([
            torch.tensor(question_embedding).float().unsqueeze(0),
            torch.tensor([[grade]], dtype=torch.float)
        ], dim=1)
        self.edge_attributes = torch.cat([self.edge_attributes, edge_attr], dim=0)
        
        # Store trust score for structural feature computation
        self.trust_scores = torch.cat([self.trust_scores, torch.tensor([grade], dtype=torch.float)], dim=0)
    
    def finalize_features(self):
        """
        Compute simple degree and trust features after all edges have been added.
        Much simpler than the previous version.
        """
        if self.edge_index.size(1) == 0:
            return  # No edges, keep zero degree features
        
        num_nodes = len(self.agents)
        
        # Compute simple degree features
        degree_features = torch.zeros(num_nodes, 4)
        
        for node_idx in range(num_nodes):
            # Get incoming and outgoing edges
            in_edges = (self.edge_index[1] == node_idx).nonzero(as_tuple=True)[0]
            out_edges = (self.edge_index[0] == node_idx).nonzero(as_tuple=True)[0]
            
            # Basic degree features
            in_degree = len(in_edges)
            out_degree = len(out_edges)
            
            # Average trust scores
            avg_in_trust = self.trust_scores[in_edges].mean().item() if in_degree > 0 else 0.0
            avg_out_trust = self.trust_scores[out_edges].mean().item() if out_degree > 0 else 0.0
            
            degree_features[node_idx] = torch.tensor([
                in_degree / num_nodes,   # Normalized in-degree
                out_degree / num_nodes,  # Normalized out-degree  
                avg_in_trust,           # Average incoming trust
                avg_out_trust           # Average outgoing trust
            ])
        
        # Update the degree part of node features (last 4 columns)
        embedding_dim = self.x.size(1) - 4
        self.x = torch.cat([self.x[:, :embedding_dim], degree_features], dim=1)
    
    def split(self, val_ratio=0.2):
        """Split edges into train and validation sets"""
        # Finalize structural features before splitting
        self.finalize_features()
        
        num_edges = self.edge_index.size(1)
        perm = torch.randperm(num_edges)
        
        val_size = int(num_edges * val_ratio)
        train_size = num_edges - val_size
        
        train_idx, val_idx = perm[:train_size], perm[train_size:]

        train_data = self.clone()
        val_data = self.clone()

        train_data.edge_index = self.edge_index[:, train_idx]
        train_data.edge_attributes = self.edge_attributes[train_idx]
        train_data.trust_scores = self.trust_scores[train_idx]
        
        val_data.edge_index = self.edge_index[:, val_idx]
        val_data.edge_attributes = self.edge_attributes[val_idx]
        val_data.trust_scores = self.trust_scores[val_idx]

        for data in [train_data, val_data]:
            data.prediction_edge_index = data.edge_index
            data.edge_trust_score = data.edge_attributes[:, -1].unsqueeze(1)
            data.edge_query_embedding = data.edge_attributes[:, :-1]
            data.prediction_source_ids = data.edge_index[0]
            data.prediction_target_ids = data.edge_index[1]
        
        return train_data, val_data


class TrustGNN(torch.nn.Module):
    def __init__(self, num_nodes, node_feature_dim, hidden_dim=32):
        super(TrustGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        
        # Much simpler architecture
        # Single GAT layer
        self.conv = GATv2Conv(
            node_feature_dim, hidden_dim,
            heads=2, concat=True,
            edge_dim=EMBEDDING_DIM + 1,
            dropout=0.1
        )
        
        # Simple edge prediction MLP for 3-class classification
        edge_input_dim = 2 * hidden_dim * 2 + EMBEDDING_DIM  # 2 nodes * hidden_dim * heads + query
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 3)  # 3 classes: low (0), medium (0.5), high (1)
        )

    def forward_logits(self, data: GraphData):
        """Forward pass returning raw logits for training"""
        if data.edge_index.size(1) == 0:
            return torch.zeros((0, 3))
        
        # Single GAT layer
        x = F.relu(self.conv(data.x, data.edge_index, data.edge_attributes))
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Edge prediction
        source_nodes_emb = x[data.prediction_edge_index[0]]
        target_nodes_emb = x[data.prediction_edge_index[1]]
        query_embeddings = data.edge_query_embedding
        
        # Concatenate for edge prediction
        edge_emb = torch.cat([source_nodes_emb, target_nodes_emb, query_embeddings], dim=1)
        
        # Return raw logits
        return self.edge_classifier(edge_emb)

    def forward(self, data: GraphData):
        # Handle empty graph case
        if data.edge_index.size(1) == 0:
            return torch.zeros(data.prediction_edge_index.size(1) if hasattr(data, 'prediction_edge_index') else 0)
        
        # Get logits and convert to trust scores
        logits = self.forward_logits(data)
        
        # Convert back to trust scores for compatibility
        # Apply softmax and compute weighted average: 0*p0 + 0.5*p1 + 1*p2
        probs = F.softmax(logits, dim=1)
        trust_scores = 0.0 * probs[:, 0] + 0.5 * probs[:, 1] + 1.0 * probs[:, 2]
        
        return trust_scores