import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from sklearn.model_selection import train_test_split

from src.config import COFNIG
from src.logged_data import load_bandit_instance


class RewardModel(nn.Module):
    def __init__(self, d, hidden_sizes=(512, 256), is_mse=True):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_out = nn.Linear(hidden_sizes[1], 1)  
        self.is_mse = is_mse

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.is_mse:
            x = self.fc_out(x)
        else:
            x = torch.sigmoid(self.fc_out(x))
        return x


class LinearRewardModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor):
        x = torch.sigmoid(self.fc1(x))
        return x

def train_model_bce(model, X_train, y_train, X_val, y_val, batch_size=512, epochs=100, lr=1e-3, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # logits + sigmoid inside

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)  # logits, shape (B,1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)

                # compute accuracy
                preds = (logits >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()
        val_loss /= len(val_dataset)
        val_acc = correct / total * 100

        # Print results
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}, Train BCE: {epoch_loss:.6f}, "
                  f"Val BCE: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")

    return model

def train_model_mse(model, X_train, y_train, X_val, y_val, batch_size=512, epochs=100, lr=1e-3, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_dataset)

        # Print results
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}, Train MSE: {epoch_loss:.6f}, Val MSE: {val_loss:.6f}")

    return model


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_model(d: int, path: str, hidden_sizes=(512, 256), is_mse=True, device=None):
    model = RewardModel(d=d, hidden_sizes=hidden_sizes, is_mse=is_mse).to(device)
    # model = LinearRewardModel(d=d).to(device)

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model weights loaded from {path}")
    return model

def train_reward_model(cfg: COFNIG):
    d, save_path, hidden_sizes, is_mse = cfg.d, cfg.reward_model_save_path, cfg.hidden_sizes, cfg.is_mse

    mu, X = load_bandit_instance(cfg.bandit_instance_path)

    if is_mse:
        y = np.sum(X * mu[0], axis=-1, keepdims=True)
    else:
        y = np.zeros((X.shape[0], X.shape[1], 1))
        y[0, :, :] = 1

    X_flat = X.reshape(-1, X.shape[2])    
    y_flat = y.reshape(-1, 1)  

    X_train, X_val, y_train, y_val = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42, shuffle=True)
    X_train = torch.from_numpy(X_train).float()
    X_val   = torch.from_numpy(X_val).float()
    y_train = torch.from_numpy(y_train).float()
    y_val   = torch.from_numpy(y_val).float()

    print("Train shape:", X_train.shape, y_train.shape)
    print("Validation shape:", X_val.shape, y_val.shape)

    model = RewardModel(d=d, hidden_sizes=(512, 256), is_mse=is_mse)
    # model = LinearRewardModel(d=d)

    if is_mse:
        model = train_model_mse(model, X_train, y_train, X_val, y_val, batch_size=1024, epochs=100, lr=1e-3)
    else:
        model = train_model_bce(model, X_train, y_train, X_val, y_val, batch_size=1024, epochs=50, lr=1e-3)

    save_model(model, save_path)



def fw0_and_grad(model: nn.Module, x: torch.Tensor):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    f_w0 = model(x).squeeze()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(f_w0, params, retain_graph=False, create_graph=False)
    grad_flat = torch.cat([g.reshape(-1) for g in grads])

    return f_w0.item(), grad_flat


def load_params_to_new_model(model: nn.Module, w_flat: torch.Tensor) -> nn.Module:
    new_model = copy.deepcopy(model)

    params = [p for p in new_model.parameters() if p.requires_grad]
    assert w_flat.numel() == sum(p.numel() for p in params), \
        f"Size mismatch: flat vector has {w_flat.numel()}, model has {sum(p.numel() for p in params)}"

    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(w_flat[offset:offset+n].view_as(p))
            offset += n

    return new_model