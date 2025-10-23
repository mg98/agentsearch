import torch
import torch.nn as nn
import torch.nn.functional as F


class RepEncoder(nn.Module):
    """Encodes one absolutely positioned token and a set of unordered tokens""" 
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim

        # positional encoding only for absolute token
        self.first_token_marker = nn.Parameter(torch.randn(hidden_dim))  # learnable marker for the first token

        self.input_proj = nn.Linear(input_dim, hidden_dim)  # (question,scores) -> embedding
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, dim_feedforward=128) # TODO feedforward dim can be tuned
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, rows):  # rows: (B: batch, N: set length, D: input_dim)
        """
        First the [:, 0, :] is the most recent question, then the rest are in arbitrary order.
        To this end, we use a positional encoding to indicate the first element.
        """
        #  B, N, D = rows.shape
        x = self.input_proj(rows)  # (B, N, hidden_dim)
        pos_enc = torch.zeros_like(x)  # (B, N, hidden_dim)
        pos_enc[:, 0, :] = self.first_token_marker  # only the first element
        x = x + pos_enc

        # transformer expects (N, B, hidden_dim)
        x = x.permute(1, 0, 2)  # (N, B, hidden_dim)
        x = self.encoder(x)  # (N, B, hidden_dim)
        
        # permutation invariant pooling
        pooled = x.mean(dim=0)  # (B, hidden_dim)
        return pooled  # (B, hidden_dim)


class ReputationPredictor(nn.Module):
    """Predicts reputation score based on agent embedding and question embedding"""
    
    def __init__(self, question_embedding, hidden_dim, n_heads, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.rep_transformer = RepEncoder(
            input_dim = question_embedding, # + 1 for the score of the question
            hidden_dim = hidden_dim,
            n_heads = n_heads,
            n_layers = n_layers
        )

        self.predict = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # self.predict = nn.Linear(hidden_dim, 1)  # final prediction layer

        
    def forward(self, tgt_x_index, x, edge_index, edge_attrs):
        """
        tgt_x_index: Index of the target node (agent), shape [1,]
        x: Node features (embeddings), shape [num_nodes, node_dim]
        edge_index: Edge indices (directed edges), shape [2, num_edges]
        edge_attrs: Edge attributes (same length as edge_index), shape [num_edges, edge_dim]
        """

        # For the tgt_x get the agents it asked questions to (outgoing edges)
        neigh_nodes = edge_index[1][edge_index[0] == tgt_x_index] # [num_neigh,]
        # print(f"number of neighbors: {neigh_nodes.shape[0]}")

        # for each neighbor
        neigh_repr = torch.zeros((neigh_nodes.shape[0], self.hidden_dim))
        for i, n in enumerate(neigh_nodes):
            # get all questions asked to n, by anyone except tgt_x
            neigh_attrs = torch.cat([ 
                edge_attrs[(edge_index[0] == tgt_x_index) & (edge_index[1] == n)],  # [1, edge_dim]
                edge_attrs[(edge_index[1] == n) & (edge_index[0] != tgt_x_index)],  # [num_neigh_of_n, edge_dim]
                ], 
                dim = 0
            ) # [num_neigh_of_n + 1, edge_dim]
        
            # agent_emb = x[n].unsqueeze(0).expand((neigh_attrs.shape[0], -1)) # [num_neigh_of_n, node_dim]

            # rows = torch.cat(
            #     [ neigh_attrs, agent_emb],
            #     dim = -1
            # ).unsqueeze(0) # [1, num_neigh_of_n + 1, edge_dim + node_dim]

            rows = neigh_attrs.unsqueeze(0) # [1, num_neigh_of_n + 1, edge_dim]

            neigh_repr[i, :] = self.rep_transformer(
                rows
            )

        # Now we have a set of neigh_repr for each neighbor node
        if neigh_repr.shape[0] > 0:
            final_repr = neigh_repr.mean(dim=0, keepdim=True) # [1, hidden_dim]
        else:
            final_repr = torch.zeros((1, self.hidden_dim))

        # return final_repr # [1, hidden_dim]

        # 3. predict
        out = F.sigmoid(self.predict(final_repr)) # [1, 1]
        return out  # [1, 1]
    

if __name__ == "__main__":
    ############### Example usage #################
    from .main import load_data_graph
    from sklearn.model_selection import train_test_split

    perc_lying = 30
    graph = load_data_graph(
        "/mnt/lourens/data/agentsearch/data",
        percentage_lying=perc_lying,
        incl_lying_label=True
    )
    x = graph.x
    edge_index = graph.edge_index
    edge_attrs = graph.edge_attr
    y = graph.y
    edge_attrs = torch.cat([edge_attrs, y.unsqueeze(-1).float()], dim=-1)

    y = graph.x_corrupt

    # take only asking agents
    asking_agents = edge_index[0].unique()
    print(f"percentage positive {y[asking_agents].sum().item()/asking_agents.shape[0]*100:.2f}%")

    train_idx, test_idx = train_test_split(
        asking_agents.numpy(),
        test_size=0.2,
        random_state=42
    )
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    model = ReputationPredictor(
        question_embedding = edge_attrs.shape[1],
        hidden_dim = 32,
        n_heads = 1,
        n_layers = 1
    )
    print(f"num of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pos_weight = torch.tensor([5.0])  # Set higher weight for positive examples
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(10):
        model.train()
        total_loss = 0
        accuracy_train = 0
        for idx in train_idx:
            optimizer.zero_grad()
            output = model(idx, x, edge_index, edge_attrs)
            label = y[idx].float().unsqueeze(0)
            loss = criterion(output.squeeze(0), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = (output > 0.5).float()
            accuracy_train += (preds.squeeze(0) == label).sum().item()


        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_idx)}, Training Accuracy: {accuracy_train/len(train_idx)}")

        model.eval()
        with torch.no_grad():
            accuracy = 0
            total_loss = 0
            for idx in test_idx:
                output = model(idx, x, edge_index, edge_attrs)
                label = y[idx].float().unsqueeze(0)
                loss = criterion(output.squeeze(0), label)
                total_loss += loss.item()

                preds = (output > 0.5).float()
                accuracy += (preds.squeeze(0) == label).sum().item()

            print(f"Testing Loss: {total_loss/len(test_idx)}, Testing Accuracy: {accuracy/len(test_idx)}")
