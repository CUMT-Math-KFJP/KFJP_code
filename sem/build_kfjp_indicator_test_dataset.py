from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_hcr_sem_dataset import TRUE_KFJP_BY_SUBJECT


# Keep data local; override KFJP_SEM_BASE to use an existing data root.
BASE = Path(os.environ.get("KFJP_SEM_BASE", Path(__file__).resolve().parent / "data"))
DATA_DIR = BASE / "4学科data"
SOURCE = BASE / "DATA" / "sem_hcr_analysis" / "hcr_sem_scholar_variables.csv"
OUT_DIR = BASE / "DATA" / "KFJP_superiority_tests"
OUT = OUT_DIR / "kfjp_indicator_test_dataset.csv"


def hhi(row: pd.Series) -> float:
    total = float(row.sum())
    if total <= 0:
        return 0.0
    shares = row[row > 0] / total
    return float((shares**2).sum())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE)
    additions = []

    for subject, kfjp_cols in TRUE_KFJP_BY_SUBJECT.items():
        matrix_path = DATA_DIR / subject / "学者-期刊矩阵" / "学者-有效期刊论文篇数矩阵.xlsx"
        pij = pd.read_excel(matrix_path, index_col=0)
        missing = sorted(set(kfjp_cols).difference(pij.columns))
        if missing:
            raise ValueError(f"{subject}: missing KFJP journals in full matrix: {missing}")

        kfjp_p = pij.reindex(columns=kfjp_cols, fill_value=0)
        subject_add = pd.DataFrame(
            {
                "subject": subject,
                "scholar": kfjp_p.index,
                "kfjp_coverage_breadth": np.log1p((kfjp_p > 0).sum(axis=1)),
                "kfjp_internal_concentration": kfjp_p.apply(hhi, axis=1),
                "kfjp_active_journal_count": (kfjp_p > 0).sum(axis=1).astype(int),
            }
        )
        additions.append(subject_add)

    add_df = pd.concat(additions, ignore_index=True)
    merged = df.merge(add_df, on=["subject", "scholar"], how="left", validate="one_to_one")

    merged["kfjp_relative_q1h_preference"] = merged["kfjp_pub"] - merged["q1h_non_kfjp_pub"]

    eps = 1e-4
    clipped_kfjp = merged["kfjp_pub"].clip(eps, 1 - eps)
    clipped_q1h = merged["q1h_non_kfjp_pub"].clip(eps, 1 - eps)
    merged["kfjp_relative_q1h_logit_preference"] = (
        np.log(clipped_kfjp / (1 - clipped_kfjp))
        - np.log(clipped_q1h / (1 - clipped_q1h))
    )

    merged.to_csv(OUT, index=False, encoding="utf-8-sig")

    summary = merged.groupby("subject").agg(
        scholars=("scholar", "count"),
        mean_kfjp_coverage_breadth=("kfjp_coverage_breadth", "mean"),
        mean_kfjp_internal_concentration=("kfjp_internal_concentration", "mean"),
        mean_kfjp_relative_q1h_preference=("kfjp_relative_q1h_preference", "mean"),
        mean_kfjp_active_journal_count=("kfjp_active_journal_count", "mean"),
    )
    summary.to_csv(OUT_DIR / "kfjp_indicator_descriptive_summary.csv", encoding="utf-8-sig")
    print(summary.round(4).to_string())
    print(f"\nSaved indicator test dataset to: {OUT}")


if __name__ == "__main__":
    main()
