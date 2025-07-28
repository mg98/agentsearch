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
        
        # Normalization statistics (will be computed after all edges are added)
        self.trust_min = None
        self.trust_max = None
        
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
        # Updated to include PageRank: in_degree, out_degree, avg_in_trust, avg_out_trust, pagerank
        degree_features = torch.zeros(num_nodes, 5)
        
        # Combine features (EMBEDDING_DIM + 5 instead of 4)
        node_features = torch.cat([embedding_features, degree_features], dim=1)
        
        return node_features
    
    def _compute_pagerank(self) -> torch.Tensor:
        """
        Compute PageRank scores for all nodes using trust scores as edge weights.
        Returns tensor of PageRank scores for each node.
        """
        if self.edge_index.size(1) == 0:
            # No edges, return uniform PageRank scores
            return torch.ones(len(self.agents)) / len(self.agents)
        
        # Build NetworkX graph with trust scores as weights
        G = nx.DiGraph()
        G.add_nodes_from(range(len(self.agents)))
        
        # Add edges with trust scores as weights
        edge_list = self.edge_index.t().numpy()  # Convert to [num_edges, 2]
        trust_weights = self.trust_scores.numpy()
        
        for i, (source, target) in enumerate(edge_list):
            # Use trust score as edge weight (higher trust = higher weight)
            weight = trust_weights[i]
            if G.has_edge(source, target):
                # If edge already exists, average the weights
                G[source][target]['weight'] = (G[source][target]['weight'] + weight) / 2
            else:
                G.add_edge(source, target, weight=weight)
        
        # Compute PageRank with trust-weighted edges
        try:
            pagerank_dict = nx.pagerank(G, weight='weight', alpha=0.85, max_iter=100, tol=1e-6)
            pagerank_scores = torch.tensor([pagerank_dict[i] for i in range(len(self.agents))], dtype=torch.float32)
        except:
            # Fallback to uniform scores if PageRank fails
            print("Warning: PageRank computation failed, using uniform scores")
            pagerank_scores = torch.ones(len(self.agents)) / len(self.agents)
        
        return pagerank_scores
    

    
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
    
    def normalize_trust_scores(self):
        """Normalize trust scores to [0, 1] range and store normalization parameters"""
        if len(self.trust_scores) == 0:
            return
        
        self.trust_min = self.trust_scores.min().item()
        self.trust_max = self.trust_scores.max().item()
        
        # Avoid division by zero
        if self.trust_max == self.trust_min:
            self.trust_scores = torch.ones_like(self.trust_scores) * 0.5
            # Update edge attributes as well
            self.edge_attributes[:, -1] = 0.5
        else:
            # Normalize to [0, 1]
            self.trust_scores = (self.trust_scores - self.trust_min) / (self.trust_max - self.trust_min)
            # Update edge attributes as well
            self.edge_attributes[:, -1] = self.trust_scores
        
        print(f"Normalized trust scores: min={self.trust_min}, max={self.trust_max}")
    
    def denormalize_trust_score(self, normalized_score: float) -> float:
        """Convert normalized score back to original scale"""
        if self.trust_min is None or self.trust_max is None:
            return normalized_score
        
        if self.trust_max == self.trust_min:
            return self.trust_min
        
        return normalized_score * (self.trust_max - self.trust_min) + self.trust_min
    
    def finalize_features(self):
        """
        Normalize trust scores and compute degree, trust, and PageRank features after all edges have been added.
        Enhanced with PageRank for authority ranking.
        """
        # First normalize trust scores
        self.normalize_trust_scores()
        
        if self.edge_index.size(1) == 0:
            return  # No edges, keep zero degree features
        
        num_nodes = len(self.agents)
        
        # Compute PageRank scores
        pagerank_scores = self._compute_pagerank()
        
        # Compute enhanced degree features including PageRank
        degree_features = torch.zeros(num_nodes, 5)
        
        for node_idx in range(num_nodes):
            # Get incoming and outgoing edges
            in_edges = (self.edge_index[1] == node_idx).nonzero(as_tuple=True)[0]
            out_edges = (self.edge_index[0] == node_idx).nonzero(as_tuple=True)[0]
            
            # Basic degree features
            in_degree = len(in_edges)
            out_degree = len(out_edges)
            
            # Average trust scores (now normalized)
            avg_in_trust = self.trust_scores[in_edges].mean().item() if in_degree > 0 else 0.0
            avg_out_trust = self.trust_scores[out_edges].mean().item() if out_degree > 0 else 0.0
            
            degree_features[node_idx] = torch.tensor([
                in_degree / num_nodes,   # Normalized in-degree
                out_degree / num_nodes,  # Normalized out-degree  
                avg_in_trust,           # Average incoming trust (normalized)
                avg_out_trust,          # Average outgoing trust (normalized)
                pagerank_scores[node_idx].item()  # PageRank authority score
            ])
        
        # Update the degree part of node features (last 5 columns now)
        embedding_dim = self.x.size(1) - 5
        self.x = torch.cat([self.x[:, :embedding_dim], degree_features], dim=1)
        
        print(f"Enhanced features computed: PageRank scores range [{pagerank_scores.min():.4f}, {pagerank_scores.max():.4f}]")
    
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
    def __init__(self, num_nodes, node_feature_dim, hidden_dim=128):
        super(TrustGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        
        # Extract just the agent embedding part (remove the 5 degree features)
        self.agent_embedding_dim = node_feature_dim - 5
        
        # Direct semantic similarity features
        self.semantic_transform = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim // 4),  # Semantic similarity feature
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        # Query-Agent interaction networks (simplified)
        self.query_transform = torch.nn.Sequential(
            torch.nn.Linear(EMBEDDING_DIM, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        # Use EMBEDDING_DIM instead of agent_embedding_dim to handle dimension adjustments
        self.target_agent_transform = torch.nn.Sequential(
            torch.nn.Linear(EMBEDDING_DIM, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        # Combine semantic similarity + contextual features
        self.prediction_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim // 4, 128),  # semantic + query + target
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1)
        )

    def forward(self, data: GraphData):
        """Forward pass with direct semantic similarity features"""
        if data.prediction_edge_index.size(1) == 0:
            return torch.zeros((0,))
        
        # Get target agent embeddings (exclude degree features)
        target_indices = data.prediction_edge_index[1]
        target_agent_emb = data.x[target_indices, :self.agent_embedding_dim]  # [num_edges, agent_emb_dim]
        query_emb = data.edge_query_embedding  # [num_edges, EMBEDDING_DIM]
        
        # Handle dimension mismatch - pad or truncate agent embeddings to match query dimension
        if target_agent_emb.size(1) != query_emb.size(1):
            if target_agent_emb.size(1) < query_emb.size(1):
                # Pad agent embeddings with zeros
                padding_size = query_emb.size(1) - target_agent_emb.size(1)
                padding = torch.zeros(target_agent_emb.size(0), padding_size, device=target_agent_emb.device)
                target_agent_emb = torch.cat([target_agent_emb, padding], dim=1)
            else:
                # Truncate agent embeddings
                target_agent_emb = target_agent_emb[:, :query_emb.size(1)]
        
        # Compute direct semantic similarity (cosine similarity)
        target_agent_emb_norm = F.normalize(target_agent_emb, p=2, dim=1)
        query_emb_norm = F.normalize(query_emb, p=2, dim=1)
        semantic_similarity = torch.sum(target_agent_emb_norm * query_emb_norm, dim=1, keepdim=True)  # [num_edges, 1]
        
        # Transform features
        semantic_features = self.semantic_transform(semantic_similarity)    # [num_edges, hidden_dim//4]
        query_features = self.query_transform(query_emb)                   # [num_edges, hidden_dim//2]
        target_features = self.target_agent_transform(target_agent_emb)    # [num_edges, hidden_dim//2]
        
        # Combine all features
        combined = torch.cat([semantic_features, query_features, target_features], dim=1)
        
        # Predict trust score with more direct mapping
        raw_scores = self.prediction_net(combined).squeeze(-1)
        
        # Use sigmoid to map to [0,1] without complex normalization
        trust_scores = torch.sigmoid(raw_scores)
        
        return trust_scores