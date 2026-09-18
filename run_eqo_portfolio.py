#!/usr/bin/env python3
"""Run unified EQO portfolio-size selection and portfolio optimization."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path, encoding, vector=False):
    try:
        df = pd.read_csv(path, index_col=0, encoding=encoding)
        values = df.values.astype(float)
        names = [str(c) for c in df.columns]
        return (values.reshape(-1), names) if vector else (values, names)
    except Exception:
        values = np.loadtxt(path, delimiter=",")
        values = np.asarray(values, dtype=float)
        if vector:
            return values.reshape(-1), []
        values = values.reshape(-1, 1) if values.ndim == 1 else values
        return values, [f"J{i}" for i in range(values.shape[1])]


def journal_names(path, fallback, encoding):
    if path is None:
        return fallback
    df = pd.read_csv(path, encoding=encoding)
    names = df.iloc[:, 0].astype(str).tolist() if df.shape[1] == 1 else df.columns.astype(str).tolist()
    if len(names) != len(fallback):
        raise ValueError("journal-name count does not match matrix columns")
    return names


def centered_matrices(M, y):
    M0 = M - M.mean(axis=0)
    y0 = y - y.mean()
    return M0.T @ M0, M0.T @ np.outer(y0, y0) @ M0


def eq_score(ind, P, Q):
    denom = float(ind.T @ P @ ind)
    return -1e18 if ind.sum() == 0 or denom <= 1e-12 else float(ind.T @ Q @ ind) / denom


def repair(ind, rng, k):
    ones, zeros = np.flatnonzero(ind).tolist(), np.flatnonzero(1 - ind).tolist()
    while len(ones) > k:
        i = rng.choice(ones); ind[i] = 0; ones.remove(i); zeros.append(i)
    while len(ones) < k:
        i = rng.choice(zeros); ind[i] = 1; zeros.remove(i); ones.append(i)
    return ind


def random_individual(n, k, rng):
    ind = np.zeros(n, dtype=int)
    ind[rng.sample(range(n), k)] = 1
    return ind


def evolve(P, Q, n, k, rng, pop_size, generations, mutation_prob):
    pop = [random_individual(n, k, rng) for _ in range(pop_size)]
    best, best_fit = pop[0].copy(), -1e18
    for _ in range(generations):
        scored = sorted(((eq_score(ind, P, Q), ind) for ind in pop), key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_fit:
            best_fit, best = scored[0][0], scored[0][1].copy()
        new_pop = [ind.copy() for _, ind in scored[:max(2, pop_size // 10)]]
        while len(new_pop) < pop_size:
            parents = [max(rng.sample(scored, min(3, len(scored))), key=lambda x: x[0])[1] for _ in range(2)]
            cut = rng.randrange(1, n) if n > 1 else 1
            child = np.concatenate([parents[0][:cut], parents[1][cut:]]).astype(int)
            flips = np.array([rng.random() < mutation_prob for _ in range(n)])
            child[flips] = 1 - child[flips]
            new_pop.append(repair(child, rng, k))
        pop = new_pop
    return best, best_fit


def linear_r2(X_train, y_train, X_test, y_test):
    X_train = X_train if X_train.ndim > 1 else X_train[:, None]
    X_test = X_test if X_test.ndim > 1 else X_test[:, None]
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(X_train)), X_train], y_train, rcond=None)
    pred = np.c_[np.ones(len(X_test)), X_test] @ beta
    ss_res = float(((y_test - pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    return 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def fit_aic(M, y, indices):
    X = M[:, indices]
    X = X if X.ndim > 1 else X[:, None]
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)
    rss = max(float(((y - np.c_[np.ones(len(X)), X] @ beta) ** 2).sum()), 1e-12)
    return rss, len(y) * math.log(rss / len(y)) + 2 * len(indices)


def add_scores(selected, weight, journal_score, pair_score, combo_score):
    for i in selected:
        journal_score[i] += weight
    for a, i in enumerate(selected):
        for j in selected[a + 1:]:
            pair_score[(i, j)] += weight
    combo_score[",".join(map(str, selected))] += weight


def search_k(M, y, k, rng, args, journal_score, pair_score, combo_score):
    n_samples, n_journals = M.shape
    local_combos, r2s = defaultdict(float), []
    for _ in range(args.runs_per_k):
        idx = list(range(n_samples)); rng.shuffle(idx)
        split = max(1, n_samples // 2)
        train, test = idx[:split], idx[split:]
        P, Q = centered_matrices(M[train], y[train])
        ind, _ = evolve(P, Q, n_journals, k, rng, args.pop_size, args.generations, args.mutation_prob)
        selected = np.flatnonzero(ind).tolist()
        if not selected or not test:
            continue
        r2 = linear_r2(M[train][:, selected], y[train], M[test][:, selected], y[test])
        r2s.append(r2)
        add_scores(selected, r2, journal_score, pair_score, combo_score)
        local_combos[",".join(map(str, selected))] += r2

    if local_combos:
        combo, combo_weight = max(local_combos.items(), key=lambda x: x[1])
        selected = [int(i) for i in combo.split(",")]
    else:
        P, Q = centered_matrices(M, y)
        ind, combo_weight = evolve(P, Q, n_journals, k, rng, args.pop_size, args.generations, args.mutation_prob)
        selected = np.flatnonzero(ind).tolist()

    P, Q = centered_matrices(M, y)
    ind = np.zeros(n_journals, dtype=int); ind[selected] = 1
    rss, aic = fit_aic(M, y, selected)
    return {
        "k": k, "indices": selected, "aic": aic, "rss": rss,
        "full_data_eq": eq_score(ind, P, Q), "best_combo_score": combo_weight,
        "mean_heldout_r2": float(np.mean(r2s)) if r2s else np.nan,
        "median_heldout_r2": float(np.median(r2s)) if r2s else np.nan,
    }


def save_vector(path, scores, names, value_col):
    pd.DataFrame({"journal_index": range(len(scores)), "journal": names, value_col: scores}) \
        .sort_values(value_col, ascending=False).to_csv(path, index=False, encoding="utf-8-sig")


def save_pairs(path, scores, names):
    rows = [{"journal_i_index": i, "journal_j_index": j, "journal_i": names[i], "journal_j": names[j], "pair_importance": v}
            for (i, j), v in scores.items()]
    pd.DataFrame(rows).sort_values("pair_importance", ascending=False).to_csv(path, index=False, encoding="utf-8-sig")


def save_combos(path, scores, names):
    rows = []
    for combo, value in scores.items():
        idx = [int(i) for i in combo.split(",") if i]
        rows.append({"journal_indices": combo, "journals": "; ".join(names[i] for i in idx), "combo_importance": value, "size": len(idx)})
    pd.DataFrame(rows).sort_values("combo_importance", ascending=False).to_csv(path, index=False, encoding="utf-8-sig")


def parse_args():
    parser = argparse.ArgumentParser(description="Run unified EQO portfolio search.")
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--journal-names", type=Path)
    parser.add_argument("--encoding", default="gbk")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs-per-k", type=int, default=1000)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--pop-size", type=int, default=80)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--mutation-prob", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    M, inferred = read_table(args.matrix, args.encoding)
    y, _ = read_table(args.response, args.encoding, vector=True)
    names = journal_names(args.journal_names, inferred, args.encoding)
    if M.shape[0] != len(y) or M.shape[1] != len(names):
        raise ValueError("Input dimensions are inconsistent.")

    min_k, max_k = max(1, args.min_k), min(args.max_k, M.shape[1])
    if min_k > max_k:
        raise ValueError("--min-k cannot exceed --max-k.")
    (args.output / "run_config.json").write_text(json.dumps({**vars(args), "n_samples": M.shape[0], "n_journals": M.shape[1]}, indent=2, default=str), encoding="utf-8")

    journal_score = np.zeros(M.shape[1])
    pair_score, combo_score, results = defaultdict(float), defaultdict(float), []
    for k in range(min_k, max_k + 1):
        results.append(search_k(M, y, k, rng, args, journal_score, pair_score, combo_score))
    best = min(results, key=lambda x: x["aic"])

    pd.DataFrame([{**r, "journal_indices": ",".join(map(str, r["indices"])), "journals": "; ".join(names[i] for i in r["indices"])} for r in results]) \
        .drop(columns="indices").to_csv(args.output / "size_search_summary.csv", index=False, encoding="utf-8-sig")
    save_vector(args.output / "journal_importance.csv", journal_score, names, "importance")
    save_pairs(args.output / "pair_importance.csv", pair_score, names)
    save_combos(args.output / "combo_importance.csv", combo_score, names)

    final = pd.DataFrame({
        "rank": range(1, len(best["indices"]) + 1),
        "journal_index": best["indices"],
        "journal": [names[i] for i in best["indices"]],
        "best_k": best["k"], "final_aic": best["aic"], "final_rss": best["rss"],
        "final_full_data_eq": best["full_data_eq"], "best_combo_score": best["best_combo_score"],
        "mean_heldout_r2_at_best_k": best["mean_heldout_r2"],
    })
    final.to_csv(args.output / "final_portfolio.csv", index=False, encoding="utf-8-sig")

    lines = [
        f"n_samples: {M.shape[0]}", f"n_journals: {M.shape[1]}", f"candidate_k: {min_k}-{max_k}",
        f"runs_per_k: {args.runs_per_k}", f"best_k: {best['k']}", f"final_aic: {best['aic']:.6f}",
        f"final_full_data_eq: {best['full_data_eq']:.6f}", f"mean_heldout_r2_at_best_k: {best['mean_heldout_r2']:.6f}",
        "final_portfolio:", *[f"  - {names[i]} (index {i})" for i in best["indices"]],
    ]
    (args.output / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
