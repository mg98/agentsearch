import torch
import torch.nn as nn
from agentsearch.baselines.ltr.utils import FeatureVector, K_VALUES
from agentsearch.utils.globals import get_torch_device
from torch.utils.data import TensorDataset, DataLoader
import wandb

class LTRModel(nn.Module):
    def __init__(self, input_dim: int = 21, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.register_buffer('feature_min', torch.zeros(input_dim))
        self.register_buffer('feature_max', torch.ones(input_dim))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        int_indices = []
        for i, k in enumerate(K_VALUES):
            int_indices.append(3 + i * 3)
        int_indices = torch.tensor([1] + int_indices, device=x.device)
        normalized = x.clone()

        range_vals = self.feature_max[int_indices] - self.feature_min[int_indices]
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        normalized[:, int_indices] = (x[:, int_indices] - self.feature_min[int_indices]) / range_vals

        return normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits)


def feature_vector_to_tensor(fv: FeatureVector) -> torch.Tensor:
    return torch.tensor(fv.to_list(), dtype=torch.float32)


def train_model(
    x_y_data: list[tuple[FeatureVector, float]],
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    val_split: float = 0.2,
    patience: int = 7,
    use_wandb: bool = True
) -> LTRModel:
    device = get_torch_device()
    model = LTRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    if use_wandb:
        wandb.init(
            project="agentsearch-ltr",
            config={
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "val_split": val_split,
                "patience": patience,
                "model": "LTRModel",
            }
        )
        wandb.watch(model, log="all", log_freq=10)

    X = torch.stack([feature_vector_to_tensor(fv) for fv, _ in x_y_data]).to(device)
    y = torch.tensor([1.0 if score > 0 else 0.0 for _, score in x_y_data], dtype=torch.float32).to(device)

    model.feature_min = X.min(dim=0).values
    model.feature_max = X.max(dim=0).values

    n_samples = len(X)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if use_wandb:
        wandb.log({
            "train_samples": n_train,
            "val_samples": n_val,
            "positive_ratio_train": y_train.mean().item(),
            "positive_ratio_val": y_val.mean().item()
        })

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val).item()

            train_predictions = model(X_train)
            train_accuracy = ((train_predictions > 0.5) == y_train).float().mean().item()
            val_accuracy = ((val_predictions > 0.5) == y_val).float().mean().item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "epochs_without_improvement": epochs_without_improvement
            })

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
            if use_wandb:
                wandb.log({"best_val_loss": best_val_loss})
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if use_wandb:
        wandb.unwatch(model)
        wandb.finish()

    model.eval()
    return model
