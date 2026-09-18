import numpy as np
import torch
from torchdiffeq import odeint

HIDDEN_DIM = 128
NUM_LAYERS = 4
DROPOUT_RATE = 0.1
F_COMP_COEFF = 0.1
BATCH_TIME = 10.0
RTOL = 1e-3
ATOL = 1e-5
EPS = 1e-12


class ODEFunc(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Dropout(DROPOUT_RATE),
            )
        ])
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Dropout(DROPOUT_RATE),
            ))
        self.layers.append(torch.nn.Sequential(torch.nn.Linear(hidden_dim, input_dim), torch.nn.Sigmoid()))
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, t, y):
        squeeze = y.ndim == 1
        if squeeze:
            y = y.unsqueeze(0)
        x = self.layers[0](y)
        for layer in self.layers[1:-1]:
            x = torch.relu(layer(x) + x)
        out = torch.clamp(self.alpha * self.layers[-1](x), 0.0, 10.0)
        f_comp = y.sum(dim=1, keepdims=True) * torch.ones(1, self.input_dim, device=y.device) * F_COMP_COEFF
        dy = y * (out - f_comp)
        return dy.squeeze(0) if squeeze else dy


def bray_curtis_loss(pred, target):
    pred = torch.clamp(pred, 0.0, None)
    return torch.mean(torch.abs(pred - target).sum(1) / torch.abs(pred + target).sum(1).clamp_min(EPS))


def normalize_rows(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    row_sum = x.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return x / row_sum


def composition_from_counts(counts):
    presence = (counts > 0).astype(np.float32)
    counts_sum = counts.sum(axis=0, keepdims=True)
    presence_sum = presence.sum(axis=0, keepdims=True)
    counts_sum[counts_sum == 0] = 1.0
    presence_sum[presence_sum == 0] = 1.0
    return (counts / counts_sum).astype(np.float32).T, (presence / presence_sum).astype(np.float32).T


def predict(model, z, device, batch_size=128, method="dopri5"):
    t = torch.tensor([0.0, BATCH_TIME], dtype=torch.float32, device=device)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            batch = torch.as_tensor(z[start:start + batch_size], dtype=torch.float32, device=device)
            out.append(odeint(model, batch, t, method=method, rtol=RTOL, atol=ATOL)[-1].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def load_model(path, input_dim, device):
    model = ODEFunc(input_dim).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    if state and next(iter(state)).startswith("module."):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model
