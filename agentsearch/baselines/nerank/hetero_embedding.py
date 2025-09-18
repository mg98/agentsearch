import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random


class MetapathWalkGenerator:
    """Generate metapath-guided random walks for heterogeneous networks."""
    
    def __init__(self, graph_data: Dict[str, List], metapath: str = "AQRQA"):
        """
        Initialize the walk generator.
        
        Args:
            graph_data: Dictionary containing network relationships
            metapath: Metapath pattern for walks (default: "AQRQA")
                     A=Answerer, Q=Question, R=Raiser
        """
        self.graph_data = graph_data
        self.metapath = metapath
        self.build_network()
    
    def build_network(self):
        """Build adjacency lists for efficient walk generation."""
        self.adjacency = defaultdict(lambda: defaultdict(list))
        
        # Build raiser -> question relationships
        if 'raiser_questions' in self.graph_data:
            for raiser, questions in self.graph_data['raiser_questions'].items():
                for q in questions:
                    self.adjacency['R'][raiser].append(('Q', q))
                    self.adjacency['Q'][q].append(('R', raiser))
        
        # Build answerer -> question relationships  
        if 'answerer_questions' in self.graph_data:
            for answerer, questions in self.graph_data['answerer_questions'].items():
                for q in questions:
                    self.adjacency['A'][answerer].append(('Q', q))
                    self.adjacency['Q'][q].append(('A', answerer))
    
    def generate_walk(self, start_node: str, start_type: str, walk_length: int) -> List[str]:
        """
        Generate a single metapath-guided random walk.
        
        Args:
            start_node: Starting node ID
            start_type: Type of starting node ('A', 'Q', or 'R')
            walk_length: Length of the walk
            
        Returns:
            List of node IDs in the walk
        """
        walk = [start_node]
        current_node = start_node
        current_type = start_type
        
        # Find position in metapath
        metapath_idx = self.metapath.index(current_type)
        
        for _ in range(walk_length - 1):
            # Get next type from metapath
            metapath_idx = (metapath_idx + 1) % len(self.metapath)
            next_type = self.metapath[metapath_idx]
            
            # Get neighbors of the correct type
            neighbors = [(n_type, n_id) for n_type, n_id in self.adjacency[current_type].get(current_node, [])
                        if n_type == next_type]
            
            if not neighbors:
                break
                
            # Random selection
            _, next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
            current_type = next_type
            
        return walk
    
    def generate_walks(self, num_walks: int = 10, walk_length: int = 13) -> List[List[str]]:
        """
        Generate multiple walks for all nodes.
        
        Args:
            num_walks: Number of walks per node
            walk_length: Length of each walk
            
        Returns:
            List of walks
        """
        walks = []
        
        for node_type in ['A', 'Q', 'R']:
            for node in self.adjacency[node_type].keys():
                for _ in range(num_walks):
                    walk = self.generate_walk(node, node_type, walk_length)
                    if len(walk) > 1:
                        walks.append(walk)
                        
        return walks


class HeterogeneousSkipGram(nn.Module):
    """Skip-gram model with negative sampling for heterogeneous networks."""
    
    def __init__(self, num_nodes: int, embedding_dim: int = 256, num_negative: int = 3):
        """
        Initialize the Skip-gram model.
        
        Args:
            num_nodes: Total number of nodes in the network
            embedding_dim: Dimension of embeddings
            num_negative: Number of negative samples
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_negative = num_negative
        
        # Center and context embeddings
        self.center_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
    
    def forward(self, center: torch.Tensor, context: torch.Tensor, 
                negative_samples: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing loss.
        
        Args:
            center: Center node indices
            context: Context node indices  
            negative_samples: Negative sample indices
            
        Returns:
            Loss value
        """
        # Get embeddings
        center_embed = self.center_embeddings(center)
        context_embed = self.context_embeddings(context)
        neg_embed = self.context_embeddings(negative_samples)
        
        # Positive scores
        pos_scores = torch.sum(center_embed * context_embed, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_scores))
        
        # Negative scores
        neg_scores = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze()
        neg_loss = -torch.log(torch.sigmoid(-neg_scores)).sum(dim=1)
        
        return (pos_loss + neg_loss).mean()
    
    def get_embeddings(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for given node indices."""
        return self.center_embeddings(node_indices)