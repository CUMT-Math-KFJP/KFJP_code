#!/usr/bin/env python3
"""Train the neural ODE composition model used by the KFJP workflow."""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchdiffeq import odeint

from kfjp_model import ATOL, BATCH_TIME, RTOL, ODEFunc, bray_curtis_loss, composition_from_counts, normalize_rows, predict

MAX_EPOCHS = 1000
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 5e-5
PATIENCE = 200
TRAIN_RATIO = 0.8
VAL_RATIO_IN_TRAIN = 0.2
SPLIT_SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_data(data, seed):
    rng = np.random.default_rng(seed)
    cols = rng.permutation(data.shape[1])
    n_train = int(data.shape[1] * TRAIN_RATIO)
    p_train, p_test = data[:, cols[:n_train]], data[:, cols[n_train:]]
    val_n = max(1, int(p_train.shape[1] * VAL_RATIO_IN_TRAIN))
    val_cols = rng.choice(p_train.shape[1], size=val_n, replace=False)
    fit_cols = np.setdiff1d(np.arange(p_train.shape[1]), val_cols)
    return p_train, p_test, p_train[:, fit_cols], p_train[:, val_cols]


def batch(z, p, size, device):
    idx = torch.randint(0, z.shape[0], (min(size, z.shape[0]),), device=device)
    return z[idx], p[idx]


def train_model(p_fit_raw, p_val_raw, output_dir, device, max_epochs, batch_size, lr, patience):
    p_fit, z_fit = composition_from_counts(p_fit_raw)
    p_val, z_val = composition_from_counts(p_val_raw)
    p_fit = torch.from_numpy(p_fit).to(device)
    z_fit = torch.from_numpy(z_fit).to(device)
    p_val = torch.from_numpy(p_val).to(device)
    z_val = torch.from_numpy(z_val).to(device)

    model = ODEFunc(p_fit.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)
    t = torch.tensor([0.0, BATCH_TIME], dtype=torch.float32, device=device)
    best_loss, best_state, bad_epochs, history = float("inf"), None, 0, []
    model_path = output_dir / "composition_model.pth"

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        bz, bp = batch(z_fit, p_fit, batch_size, device)
        train_loss = bray_curtis_loss(odeint(model, bz, t, method="dopri5", rtol=RTOL, atol=ATOL)[-1], bp)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = bray_curtis_loss(odeint(model, z_val, t, method="dopri5", rtol=RTOL, atol=ATOL)[-1], p_val)
        val_value = float(val_loss.item())
        scheduler.step(val_value)
        history.append([epoch, float(train_loss.item()), val_value, float(optimizer.param_groups[0]["lr"])])

        if val_value <= best_loss:
            best_loss = val_value
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            torch.save(best_state, model_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}/{max_epochs} | train={train_loss.item():.6f} | val={val_value:.6f}")
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best validation loss: {best_loss:.6f}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid composition model.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    np.savetxt(output_dir / "training_history.csv", np.asarray(history), delimiter=",",
               header="epoch,train_bray_curtis_loss,val_bray_curtis_loss,lr", comments="",
               fmt=["%d", "%.8f", "%.8f", "%.8f"])
    return model, model_path, history


def main():
    parser = argparse.ArgumentParser(description="Train the neural ODE composition model.")
    parser.add_argument("--data-file", type=Path, default=Path(__file__).with_name("data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data = np.loadtxt(args.data_file.resolve(), delimiter=",", ndmin=2).astype(np.float32)
    p_train, p_test, p_fit, p_val = split_data(data, args.seed)
    print(f"Loaded data: {data.shape}; train={p_train.shape}, test={p_test.shape}")
    np.savetxt(output_dir / "P_train.csv", p_train, delimiter=",", fmt="%.6f")
    np.savetxt(output_dir / "P_test.csv", p_test, delimiter=",", fmt="%.6f")

    pall, zall = composition_from_counts(p_train)
    ptest, ztest = composition_from_counts(p_test)
    for name, array in {"pall": pall, "zall": zall, "ptst": ptest, "ztst": ztest}.items():
        np.savetxt(output_dir / f"{name}.csv", array, delimiter=",", fmt="%.6f")

    start = time.time()
    model, model_path, _ = train_model(p_fit, p_val, output_dir, device, args.max_epochs, args.batch_size, args.lr, args.patience)
    full_pred = predict(model, zall, device)
    test_pred = predict(model, ztest, device)
    np.savetxt(output_dir / "full_input_predictions.csv", full_pred, delimiter=",", fmt="%.8f")
    np.savetxt(output_dir / "test_input_predictions.csv", test_pred, delimiter=",", fmt="%.8f")
    np.savetxt(output_dir / "full_input_predictions_normalized.csv", normalize_rows(full_pred), delimiter=",", fmt="%.8f")
    np.savetxt(output_dir / "test_input_predictions_normalized.csv", normalize_rows(test_pred), delimiter=",", fmt="%.8f")

    print(f"Training seconds: {time.time() - start:.2f}")
    print(f"Saved model: {model_path}")
    print(f"Saved model inputs and predictions to: {output_dir}")
    print(f"zall shape: {zall.shape}, full-input prediction shape: {full_pred.shape}")


if __name__ == "__main__":
    main()
