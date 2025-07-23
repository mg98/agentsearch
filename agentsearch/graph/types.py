import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from agentsearch.utils.globals import EMBEDDING_DIM
from agentsearch.dataset.agents import Agent
import random
import numpy as np
from typing import List, Tuple

class GraphData(Data):
    def __init__(self, agents: list[Agent]):
        super().__init__()
        self.agents = agents
        self.agent_id_to_index = {agent.id: i for i, agent in enumerate(agents)}

        self.x = torch.eye(len(agents))  # One-hot encoding for initial node features
        
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.edge_attributes = torch.empty((0, EMBEDDING_DIM + 1))
        
    def add_edge(self, source_agent: Agent, target_agent: Agent, question_embedding: np.ndarray, grade: float):
        source_idx = self.agent_id_to_index[source_agent.id]
        target_idx = self.agent_id_to_index[target_agent.id]

        new_edge = torch.tensor([[source_idx], [target_idx]], dtype=torch.long)
        self.edge_index = torch.cat([self.edge_index, new_edge], dim=1)

        edge_attr = torch.cat([
            torch.tensor(question_embedding).float().unsqueeze(0),
            torch.tensor([[grade]], dtype=torch.float)
        ], dim=1)
        self.edge_attributes = torch.cat([self.edge_attributes, edge_attr], dim=0)

    def split(self, val_ratio=0.2):
        num_edges = self.edge_index.size(1)
        perm = torch.randperm(num_edges)
        
        val_size = int(num_edges * val_ratio)
        train_size = num_edges - val_size
        
        train_idx, val_idx = perm[:train_size], perm[train_size:]

        train_data = self.clone()
        val_data = self.clone()

        train_data.edge_index = self.edge_index[:, train_idx]
        train_data.edge_attributes = self.edge_attributes[train_idx]
        
        val_data.edge_index = self.edge_index[:, val_idx]
        val_data.edge_attributes = self.edge_attributes[val_idx]

        for data in [train_data, val_data]:
            data.prediction_edge_index = data.edge_index
            data.edge_trust_score = data.edge_attributes[:, -1].unsqueeze(1)
            data.edge_query_embedding = data.edge_attributes[:, :-1]
            data.prediction_source_ids = data.edge_index[0]
            data.prediction_target_ids = data.edge_index[1]
        
        return train_data, val_data


class TrustGNN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim=32, out_channels=16):
        super(TrustGNN, self).__init__()
        self.node_emb = torch.nn.Embedding(num_nodes, embedding_dim)
        
        # Initialize embeddings with Xavier initialization
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        
        # GNN layers with reduced capacity for small graph
        self.conv1 = GATv2Conv(embedding_dim, out_channels, heads=4, concat=True, edge_dim=EMBEDDING_DIM + 1, dropout=0.1)
        self.conv2 = GATv2Conv(out_channels * 4, out_channels, heads=2, concat=True, edge_dim=EMBEDDING_DIM + 1, dropout=0.1)
        self.conv3 = GATv2Conv(out_channels * 2, out_channels, heads=1, concat=False, edge_dim=EMBEDDING_DIM + 1, dropout=0.1)
        
        # Edge prediction MLP with reduced size
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * out_channels, 64),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 1)
            # torch.nn.Sigmoid()
        )

    def forward(self, data: GraphData):
        node_features = self.node_emb(data.x.argmax(dim=1))
        
        # Handle empty graph case
        if data.edge_index.size(1) == 0:
            # Return zeros for predictions if no edges
            return torch.zeros(data.prediction_edge_index.size(1) if hasattr(data, 'prediction_edge_index') else 0)
        
        # Message passing with residual connections
        x1 = F.elu(self.conv1(node_features, data.edge_index, data.edge_attributes))
        x2 = F.elu(self.conv2(x1, data.edge_index, data.edge_attributes))
        x3 = self.conv3(x2, data.edge_index, data.edge_attributes)
        
        # Edge prediction
        source_nodes_emb = x3[data.prediction_edge_index[0]]
        target_nodes_emb = x3[data.prediction_edge_index[1]]
        
        edge_emb = torch.cat([source_nodes_emb, target_nodes_emb], dim=1)
        
        return self.edge_mlp(edge_emb).squeeze()