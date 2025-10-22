import torch
import torch.nn as nn
from agentsearch.baselines.pointwise.utils import FeatureVector
from agentsearch.utils.globals import get_torch_device

class PointwiseModel(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.register_buffer('feature_min', torch.zeros(input_dim))
        self.register_buffer('feature_max', torch.ones(input_dim))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        range_vals = self.feature_max - self.feature_min
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        return (x - self.feature_min) / range_vals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        return self.net(x).squeeze(-1)


def feature_vector_to_tensor(fv: FeatureVector) -> torch.Tensor:
    return torch.tensor([
        fv.cosine_similarity,
        fv.num_reports,
        fv.score_min,
        fv.score_max,
        fv.score_mean,
        fv.score_variance,
        fv.topic_num_reports,
        fv.topic_score_min,
        fv.topic_score_max,
        fv.topic_score_mean,
        fv.topic_score_variance,
    ], dtype=torch.float32)


def train_model(
    x_y_data: list[tuple[FeatureVector, float]],
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32
) -> PointwiseModel:
    device = get_torch_device()
    model = PointwiseModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.stack([feature_vector_to_tensor(fv) for fv, _ in x_y_data]).to(device)
    y = torch.tensor([score for _, score in x_y_data], dtype=torch.float32).to(device)

    model.feature_min = X.min(dim=0).values
    model.feature_max = X.max(dim=0).values

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    return model
