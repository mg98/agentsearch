import torch
import torch.nn as nn
import wandb

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    

def train_epoch(model, X, y, optimizer, criterion, batch_size=32):
    model.train()
    optimizer.zero_grad()

    num_samples = X.size(0)
    permutation = torch.randperm(num_samples)
    
    tot_loss = 0
    accs = []

    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X[indices], y[indices]

        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).long()
        accuracy = (preds == batch_y).sum().item() / batch_y.size(0)

        accs.append(accuracy)
        tot_loss += loss.item()

    return tot_loss, sum(accs) / len(accs)

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        predictions = (torch.sigmoid(outputs) >= 0.5).long()
        accuracy = (predictions == y).sum().item() / y.size(0)
    return accuracy

def train_model(model, X, y, config): 
    """Train binary classification MLP"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # split data in train, val, test
    num_samples = X.shape[0]
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    test_size = num_samples - train_size - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    use_wandb = config['experiment'].get('use_wandb', False)

    if use_wandb:
        wandb.init(project=config['experiment']['project_name'], 
            name=config['experiment']['experiment_name'] + f"_lying{config['data']['percentage_lying']}",
            config=config
        )


    val_accs = []
    test_accs = []
    for epoch in range(config['training']['epochs']):  # number of epochs
        optimizer.zero_grad()

        loss, train_accs = train_epoch(model, X_train, y_train, optimizer, criterion)

        val_acc = evaluate(model, X_val, y_val)
        test_acc = evaluate(model, X_test, y_test)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        best_epoch, best_test_acc = get_best_epoch(val_accs, test_accs)

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': loss,
                'train_acc': train_accs,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'best/epoch': best_epoch,
                'best/test_acc': best_test_acc
            })

        if (epoch+1) % 10 == 0: print(f'Epoch [{epoch+1}/100], Loss: {loss:.4f}, Train Acc: {train_accs:.4f}, Val Acc: {val_acc:.4f}')
    
    print(f'Best Val Epoch: {best_epoch}, Test Accuracy: {best_test_acc:.4f}')

    if use_wandb:
        wandb.finish()

    return best_test_acc

def get_best_epoch(val_accs, test_accs):
    best_val_idx = val_accs.index(max(val_accs))
    return best_val_idx, test_accs[best_val_idx]