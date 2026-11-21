from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

import metrics

FLOOR = metrics.FLOOR

_INIT_LOCK = threading.Lock()


def _make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_cuda():
    try:
        return torch.autocast(device_type="cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


@dataclass
class TrainingConfig:
    hidden: tuple[int, ...] = (1024, 1024)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True

    def as_dict(self) -> dict:
        return {
            "hidden": list(self.hidden),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "seed": self.seed,
        }


class SpectrumMLP(nn.Module):
    def __init__(self, n_features: int, n_points: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.BatchNorm1d(size), nn.Dropout(dropout)]
            width = size
        layers += [nn.Linear(width, n_points), nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sid_loss(predicted: torch.Tensor, target: torch.Tensor, floor: float = FLOOR) -> torch.Tensor:
    p = predicted.clamp_min(0) + floor
    q = target.clamp_min(0) + floor
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    return (p * torch.log(p / q) + q * torch.log(q / p)).sum(dim=1).mean()


def _to_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        chunk = _to_tensor(X[start : start + batch_size], device)
        out.append(model(chunk).float().cpu().numpy())
    return np.clip(np.concatenate(out), 0.0, None)


@dataclass
class TrainingResult:
    model: nn.Module
    best_epoch: int
    best_score: float | None
    history: list[float] = field(default_factory=list)


def train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    config: TrainingConfig,
    X_val: np.ndarray | None = None,
    Y_val: np.ndarray | None = None,
    n_epochs: int | None = None,
    on_epoch=None,
) -> TrainingResult:
    device = config.device
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    with _INIT_LOCK:
        torch.manual_seed(config.seed)
        model = SpectrumMLP(
            X_train.shape[1], Y_train.shape[1], tuple(config.hidden), config.dropout
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    use_amp = config.use_amp and device.startswith("cuda")
    scaler = _make_grad_scaler() if use_amp else None

    X_train_t = _to_tensor(X_train, device)
    Y_train_t = _to_tensor(Y_train, device)
    n_rows = len(X_train_t)

    validating = X_val is not None and Y_val is not None
    total_epochs = n_epochs if n_epochs is not None else config.max_epochs

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    best_score, best_state, best_epoch, n_worse = -np.inf, None, 0, 0
    history: list[float] = []

    for epoch in range(total_epochs):
        model.train()
        order = torch.randperm(n_rows, generator=generator).to(device)
        n_batches = max(1, n_rows // config.batch_size)
        for batch in range(n_batches):
            rows = order[batch * config.batch_size : (batch + 1) * config.batch_size]
            if len(rows) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with _autocast_cuda():
                    loss = sid_loss(model(X_train_t[rows]), Y_train_t[rows])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = sid_loss(model(X_train_t[rows]), Y_train_t[rows])
                loss.backward()
                optimizer.step()

        if not validating:
            continue

        score = -metrics.sid(Y_val, predict(model, X_val, device))
        history.append(score)
        if on_epoch is not None:
            on_epoch(epoch, score)

        if score > best_score + 1e-6:
            best_score, best_epoch, n_worse = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            n_worse += 1
            if config.patience and n_worse >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        best_epoch=best_epoch if validating else total_epochs,
        best_score=best_score if validating else None,
        history=history,
    )
