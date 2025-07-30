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
import math

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
        
        # Add simple structural features (will be updated after edges are added)
        # Updated to include: degree (5) + transitive trust (3) = 8 total structural features
        # [in_degree, out_degree, avg_in_trust, avg_out_trust, pagerank, endorsement_score, two_hop_trust, trust_consistency]
        structural_features = torch.zeros(num_nodes, 8)
        
        # Combine features (EMBEDDING_DIM + 8 structural features)
        node_features = torch.cat([embedding_features, structural_features], dim=1)
        
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
    
    def _compute_transitive_trust_features(self) -> torch.Tensor:
        """
        Compute transitive trust features specifically for 2-hop star topology.
        Returns tensor of transitive trust features for each node.
        """
        num_nodes = len(self.agents)
        
        # Initialize transitive features: [endorsement_score, two_hop_trust, trust_consistency]
        transitive_features = torch.zeros(num_nodes, 3)
        
        if self.edge_index.size(1) == 0:
            return transitive_features
        
        # Build adjacency matrix with trust weights
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        edge_list = self.edge_index.t()
        
        for i, (source, target) in enumerate(edge_list):
            adj_matrix[source, target] = self.trust_scores[i]
        
        # 1. Endorsement Score: How many high-trust agents trust this agent
        for target_idx in range(num_nodes):
            incoming_edges = self.edge_index[1] == target_idx
            if incoming_edges.any():
                source_indices = self.edge_index[0][incoming_edges]
                trust_scores = self.trust_scores[incoming_edges]
                
                # Weight endorsements by the trustworthiness of endorsers
                endorser_trustworthiness = []
                for source_idx in source_indices:
                    # How much trust does this endorser receive from others?
                    endorser_incoming = (self.edge_index[1] == source_idx)
                    if endorser_incoming.any():
                        avg_trust_to_endorser = self.trust_scores[endorser_incoming].mean().item()
                    else:
                        avg_trust_to_endorser = 0.5  # Neutral if no incoming trust
                    endorser_trustworthiness.append(avg_trust_to_endorser)
                
                endorser_weights = torch.tensor(endorser_trustworthiness)
                weighted_endorsement = (trust_scores * endorser_weights).sum().item()
                transitive_features[target_idx, 0] = weighted_endorsement
        
        # 2. Two-hop Trust Propagation: A->B->C, what's the implied A->C trust?
        two_hop_trust = torch.matmul(adj_matrix, adj_matrix)  # A*A gives 2-hop paths
        
        for node_idx in range(num_nodes):
            # Average 2-hop trust received by this node
            two_hop_received = two_hop_trust[:, node_idx]
            non_zero_paths = two_hop_received[two_hop_received > 0]
            if len(non_zero_paths) > 0:
                transitive_features[node_idx, 1] = non_zero_paths.mean().item()
        
        # 3. Trust Consistency: How consistent are the trust scores involving this agent
        for node_idx in range(num_nodes):
            # Get all trust scores where this node is involved (as source or target)
            as_source = (self.edge_index[0] == node_idx)
            as_target = (self.edge_index[1] == node_idx)
            
            all_scores = torch.cat([
                self.trust_scores[as_source] if as_source.any() else torch.empty(0),
                self.trust_scores[as_target] if as_target.any() else torch.empty(0)
            ])
            
            if len(all_scores) > 1:
                # Higher consistency = lower variance = higher score
                consistency = 1.0 / (1.0 + all_scores.var().item())
                transitive_features[node_idx, 2] = consistency
            else:
                transitive_features[node_idx, 2] = 0.5  # Neutral if insufficient data
        
        return transitive_features

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
        """Normalize trust scores to [0, 1] range using logarithmic scaling"""
        if len(self.trust_scores) == 0:
            return
        
        self.trust_min = self.trust_scores.min().item()
        self.trust_max = self.trust_scores.max().item()
        
        # Store original values before normalization for evaluation
        self.original_trust_min = self.trust_min
        self.original_trust_max = self.trust_max
        
        # Avoid division by zero or log(0)
        if self.trust_max == self.trust_min:
            self.trust_scores = torch.ones_like(self.trust_scores) * 0.5
            # Update edge attributes as well
            self.edge_attributes[:, -1] = 0.5
        else:
            # Apply log(1 + x) transformation and normalize to [0, 1]
            log_scores = torch.log1p(self.trust_scores)
            log_min = torch.log1p(torch.tensor(self.trust_min))
            log_max = torch.log1p(torch.tensor(self.trust_max))
            
            # Normalize log scores to [0, 1]
            self.trust_scores = (log_scores - log_min) / (log_max - log_min)
            # Update edge attributes as well
            self.edge_attributes[:, -1] = self.trust_scores
        
        print(f"Normalized trust scores: min={self.trust_min}, max={self.trust_max}")
    
    def denormalize_trust_score(self, normalized_score: float) -> float:
        """Convert normalized score back to original scale using exponential"""
        if self.trust_min is None or self.trust_max is None:
            return normalized_score
        
        if self.trust_max == self.trust_min:
            return self.trust_min
        
        # Convert back from [0,1] to log space
        log_min = math.log1p(self.trust_min)
        log_max = math.log1p(self.trust_max)
        log_score = normalized_score * (log_max - log_min) + log_min
        
        # Convert from log space back to original scale
        return math.expm1(log_score)
    
    def normalize_single_score(self, raw_score: float) -> float:
        """Normalize a single raw score using the same logic as normalize_trust_scores"""
        if self.original_trust_min is None or self.original_trust_max is None:
            return raw_score
        
        if self.original_trust_max == self.original_trust_min:
            return 0.5
        
        # Apply log(1 + x) transformation and normalize to [0, 1]
        log_raw = math.log1p(raw_score)
        log_min = math.log1p(self.original_trust_min)
        log_max = math.log1p(self.original_trust_max)
        
        # Normalize log score to [0, 1]
        normalized = (log_raw - log_min) / (log_max - log_min)
        return normalized
    
    def finalize_features(self):
        """
        Normalize trust scores and compute degree, trust, PageRank, and transitive trust features 
        after all edges have been added. Enhanced with transitive trust patterns for 2-hop topology.
        """
        # First normalize trust scores
        self.normalize_trust_scores()
        
        if self.edge_index.size(1) == 0:
            return  # No edges, keep zero degree features
        
        num_nodes = len(self.agents)
        
        # Compute PageRank scores
        pagerank_scores = self._compute_pagerank()
        
        # Compute transitive trust features
        transitive_features = self._compute_transitive_trust_features()
        
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
        
        # Combine all structural features: degree (5) + transitive (3) = 8 total
        structural_features = torch.cat([degree_features, transitive_features], dim=1)
        
        # Update the structural part of node features (last 8 columns now)
        embedding_dim = self.x.size(1) - 8  # Now 8 structural features
        self.x = torch.cat([self.x[:, :embedding_dim], structural_features], dim=1)
        
        print(f"Enhanced features computed: PageRank range [{pagerank_scores.min():.4f}, {pagerank_scores.max():.4f}]")
        print(f"Transitive features computed: Endorsement range [{transitive_features[:, 0].min():.4f}, {transitive_features[:, 0].max():.4f}]")
        print(f"Two-hop trust range [{transitive_features[:, 1].min():.4f}, {transitive_features[:, 1].max():.4f}]")
    
    def split(self, val_ratio=0.2):
        """Split edges into train and validation sets"""
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
        
        # Extract just the agent embedding part (remove the 8 structural features)
        self.agent_embedding_dim = node_feature_dim - 8
        self.structural_feature_dim = 8
        
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
        
        # Transform structural features (including transitive trust features)
        self.structural_transform = torch.nn.Sequential(
            torch.nn.Linear(self.structural_feature_dim, hidden_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        # Combine semantic similarity + contextual features + structural features
        self.prediction_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim // 4 + hidden_dim // 4, 128),  # semantic + query + target + structural
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
        
        # Get target agent embeddings and structural features separately
        target_indices = data.prediction_edge_index[1]
        target_agent_emb = data.x[target_indices, :self.agent_embedding_dim]  # [num_edges, agent_emb_dim]
        target_structural = data.x[target_indices, self.agent_embedding_dim:]  # [num_edges, structural_dim]
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
        structural_features = self.structural_transform(target_structural) # [num_edges, hidden_dim//4]
        
        # Combine all features including transitive trust
        combined = torch.cat([semantic_features, query_features, target_features, structural_features], dim=1)
        
        # Predict trust score with more direct mapping
        raw_scores = self.prediction_net(combined).squeeze(-1)
        
        # Use sigmoid to map to [0,1] without complex normalization
        trust_scores = torch.sigmoid(raw_scores)
        
        return trust_scores