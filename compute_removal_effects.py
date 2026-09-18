#!/usr/bin/env python3
"""Compute journal removal effects from trained composition-model outputs."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint
from tqdm import tqdm

from kfjp_model import ATOL, RTOL, load_model

T_END = 10.0
STEP_SIZE = 0.2


def norm(x):
    total = x.sum()
    return x / total if total > 0 else x


def compute_effects(zall, full_pred, model, t_grid, device, batch_size):
    zall = torch.as_tensor(zall, dtype=torch.float32, device=device)
    full_pred = torch.as_tensor(full_pred, dtype=torch.float32, device=device)
    n_scholars, n_journals = zall.shape
    effects = np.zeros((n_scholars, n_journals), dtype=np.float64)

    with torch.no_grad():
        for s in tqdm(range(n_scholars), desc="Computing removal effects"):
            z = zall[s]
            removed_inputs = []
            for j in range(n_journals):
                z_removed = z.clone()
                z_removed[j] = 0.0
                removed_inputs.append(norm(z_removed))

            removed_pred = []
            for start in range(0, n_journals, batch_size):
                batch = torch.stack(removed_inputs[start:start + batch_size])
                removed_pred.append(odeint(model, batch, t_grid, method="euler", rtol=RTOL, atol=ATOL)[-1])
            removed_pred = torch.cat(removed_pred)

            z_total = z.sum().item()
            for j in range(n_journals):
                delta = torch.abs(full_pred[s] - removed_pred[j]).sum().item()
                frequency = z[j].item() / z_total if z_total > 0 else 0.0
                effects[s, j] = delta * (1.0 - frequency)

    return effects, np.median(effects, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Compute journal removal-effect matrix.")
    parser.add_argument("--input-dir", type=Path, default=Path.cwd())
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--t-end", type=float, default=T_END)
    parser.add_argument("--step-size", type=float, default=STEP_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    zall = np.loadtxt(input_dir / "zall.csv", delimiter=",", ndmin=2)
    full_pred = np.loadtxt(input_dir / "full_input_predictions.csv", delimiter=",", ndmin=2)
    if zall.shape != full_pred.shape:
        raise ValueError(f"zall shape {zall.shape} does not match full_input_predictions shape {full_pred.shape}.")

    model_path = (args.model or input_dir / "composition_model.pth").resolve()
    model = load_model(model_path, input_dim=zall.shape[1], device=device)
    t_grid = torch.arange(0.0, args.t_end + args.step_size / 2.0, args.step_size, dtype=torch.float32, device=device)
    matrix, journal_effect = compute_effects(zall, full_pred, model, t_grid, device, args.batch_size)

    np.savetxt(output_dir / "removal_effect_matrix.csv", matrix, delimiter=",", fmt="%.6f")
    np.savetxt(output_dir / "journal_removal_effect.csv", journal_effect, delimiter=",", fmt="%.6f")
    print(f"Saved removal-effect matrix: {output_dir / 'removal_effect_matrix.csv'}")
    print(f"Saved journal-level effects: {output_dir / 'journal_removal_effect.csv'}")
    print(f"effect_matrix shape: {matrix.shape}")


if __name__ == "__main__":
    main()
