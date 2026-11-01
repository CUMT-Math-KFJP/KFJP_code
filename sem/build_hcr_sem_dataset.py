from __future__ import annotations

import math
import re
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Workbook contains no default style")


# Keep data local; override KFJP_SEM_BASE to use an existing data root.
BASE = Path(os.environ.get("KFJP_SEM_BASE", Path(__file__).resolve().parent / "data"))
DATA_DIR = BASE / "4学科data"
PARTITION_PATH = BASE / "DATA" / "期刊分区.xlsx"
OUT_DIR = BASE / "DATA" / "sem_hcr_analysis"
SUBJECTS = ["phy", "che", "bio", "mat"]
CURRENT_YEAR = 2026

TRUE_KFJP_BY_SUBJECT = {
    "phy": [
        "ADVANCED MATERIALS",
        "ANGEWANDTE CHEMIE-INTERNATIONAL EDITION",
        "CHEMICAL COMMUNICATIONS",
        "JOURNAL OF MATERIALS CHEMISTRY C",
        "LIGHT-SCIENCE & APPLICATIONS",
        "NATURE PHOTONICS",
        "NATURE PHYSICS",
        "NATURE REVIEWS PHYSICS",
        "SMALL",
        "SOLID STATE COMMUNICATIONS",
    ],
    "che": [
        "ADVANCED MATERIALS",
        "ADVANCED ENERGY MATERIALS",
        "CHEMICAL COMMUNICATIONS",
        "CHEMPHYSCHEM",
        "JOURNAL OF CHEMICAL PHYSICS",
        "JOURNAL OF PHYSICAL CHEMISTRY B",
        "MICROPOROUS AND MESOPOROUS MATERIALS",
        "NATURE MATERIALS",
        "NATURE PHOTONICS",
        "PHYSICAL REVIEW LETTERS",
        "PROGRESS IN POLYMER SCIENCE",
    ],
    "bio": [
        "JOURNAL OF EXPERIMENTAL MEDICINE",
        "MOLECULAR AND CELLULAR BIOLOGY",
        "SCIENCE SIGNALING",
        "NATURE GENETICS",
        "GENES & DEVELOPMENT",
        "JOURNAL OF CELL BIOLOGY",
        "NATURE REVIEWS MICROBIOLOGY",
        "MICROBIOME",
        "NATURE REVIEWS GENETICS",
        "GASTROENTEROLOGY",
    ],
    "mat": [
        "JOURNAL OF COLLOID AND INTERFACE SCIENCE",
        "JOURNAL OF APPLIED PHYSICS",
        "COORDINATION CHEMISTRY REVIEWS",
        "CHEMICAL PHYSICS LETTERS",
        "JOURNAL OF MATERIALS CHEMISTRY",
        "CHEMICAL COMMUNICATIONS",
        "REVIEWS OF MODERN PHYSICS",
        "MATERIALS HORIZONS",
        "NATURE NANOTECHNOLOGY",
        "SOLID STATE COMMUNICATIONS",
        "CHEMICAL SCIENCE",
        "NATURE CHEMISTRY",
    ],
}


def norm_journal(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().upper())


def share_entropy(values: pd.Series) -> float:
    total = float(values.sum())
    if total <= 0:
        return np.nan
    shares = values[values > 0] / total
    return float(-(shares * np.log(shares)).sum())


def hhi(values: pd.Series) -> float:
    total = float(values.sum())
    if total <= 0:
        return np.nan
    shares = values[values > 0] / total
    return float((shares**2).sum())


def topn_share(values: pd.Series, n: int = 5) -> float:
    total = float(values.sum())
    if total <= 0:
        return np.nan
    return float(values.sort_values(ascending=False).head(n).sum() / total)


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den and den > 0 else np.nan


def parse_raw_profile(path: Path) -> dict[str, float | int | str]:
    df = pd.read_excel(path, sheet_name=0, header=None)
    first_col = df.iloc[:, 0].astype(str).str.strip()

    def meta_value(label: str) -> float:
        idx = first_col[first_col.str.lower() == label.lower()].index
        if len(idx) == 0 or df.shape[1] < 2:
            return np.nan
        return pd.to_numeric(df.iloc[idx[0], 1], errors="coerce")

    h_index = meta_value("h-index")
    results_found = meta_value("Results found")
    sum_times_cited = meta_value("Sum of the Times Cited")
    avg_citations = meta_value("Average Citations per Item")

    title_idx = first_col[first_col == "Title"].index
    pub_years = pd.Series(dtype="float64")
    if len(title_idx) > 0:
        header_row = title_idx[0]
        records = df.iloc[header_row + 1 :].copy()
        records.columns = df.iloc[header_row].tolist()
        if "Publication Year" in records.columns:
            pub_years = pd.to_numeric(records["Publication Year"], errors="coerce").dropna()

    first_year = int(pub_years.min()) if len(pub_years) else np.nan
    last_year = int(pub_years.max()) if len(pub_years) else np.nan
    active_years = int(pub_years.nunique()) if len(pub_years) else np.nan
    academic_age = CURRENT_YEAR - first_year + 1 if not pd.isna(first_year) else np.nan

    return {
        "scholar": path.stem,
        "raw_results_found": results_found,
        "raw_sum_times_cited": sum_times_cited,
        "raw_avg_citations_per_item": avg_citations,
        "h_index": h_index,
        "first_publication_year": first_year,
        "last_publication_year": last_year,
        "active_publication_years": active_years,
        "academic_age": academic_age,
    }


def load_partition() -> pd.DataFrame:
    partition = pd.read_excel(PARTITION_PATH)
    partition["journal_norm"] = partition["NAME"].map(norm_journal)
    partition["quartile_norm"] = partition["Quartile"].astype(str).str.upper().str.replace(" ", "", regex=False)
    partition = partition.drop_duplicates("journal_norm", keep="first")
    return partition[["journal_norm", "quartile_norm"]]


def build_subject(subject: str, partition: pd.DataFrame) -> pd.DataFrame:
    subject_dir = DATA_DIR / subject
    matrix_dir = subject_dir / "学者-期刊矩阵"
    filter_dir = subject_dir / "百分之几筛选结果"

    pij = pd.read_excel(matrix_dir / "学者-有效期刊论文篇数矩阵.xlsx", index_col=0)
    cij = pd.read_excel(matrix_dir / "学者-有效期刊被引次数矩阵.xlsx", index_col=0)
    pij_kfjp = pd.read_excel(filter_dir / "P_ij_筛选后.xlsx", index_col=0)
    cij_kfjp = pd.read_excel(filter_dir / "C_ij_筛选后.xlsx", index_col=0)

    if not pij.index.equals(cij.index) or not pij.columns.equals(cij.columns):
        raise ValueError(f"{subject}: P_ij and C_ij are not aligned")

    journal_quartile = (
        pd.DataFrame({"journal_norm": [norm_journal(col) for col in pij.columns]}, index=pij.columns)
        .merge(partition, how="left", on="journal_norm")
        .set_index(pij.columns)
    )
    q = journal_quartile["quartile_norm"]
    q1h_cols = q.index[q == "Q1H"]
    q1plus_cols = q.index[q.isin(["Q1H", "Q1"])]
    known_cols = q.index[q.notna()]
    candidate_cols = pd.Index(pij_kfjp.columns)
    true_kfjp = pd.Index(TRUE_KFJP_BY_SUBJECT[subject])
    missing_kfjp = true_kfjp.difference(candidate_cols)
    if len(missing_kfjp) > 0:
        raise ValueError(f"{subject}: true KFJP journals missing from filtered candidates: {missing_kfjp.tolist()}")
    kfjp_cols = true_kfjp
    non_kfjp_cols = pij.columns.difference(kfjp_cols)
    q1h_non_kfjp_cols = q1h_cols.difference(kfjp_cols)

    raw_profiles = {
        item["scholar"]: item
        for item in (parse_raw_profile(path) for path in subject_dir.glob("*.xlsx"))
    }

    rows = []
    for scholar in pij.index:
        p = pij.loc[scholar]
        c = cij.loc[scholar]
        pk = p.reindex(kfjp_cols, fill_value=0)
        ck = c.reindex(kfjp_cols, fill_value=0)

        total_papers = float(p.sum())
        total_citations = float(c.sum())
        journal_count = int((p > 0).sum())
        kfjp_papers = float(pk.sum())
        kfjp_citations = float(ck.sum())
        non_kfjp_papers = float(p.reindex(non_kfjp_cols, fill_value=0).sum())
        non_kfjp_citations = float(c.reindex(non_kfjp_cols, fill_value=0).sum())
        q1h_papers = float(p.reindex(q1h_cols, fill_value=0).sum())
        q1h_citations = float(c.reindex(q1h_cols, fill_value=0).sum())
        q1h_non_kfjp_papers = float(p.reindex(q1h_non_kfjp_cols, fill_value=0).sum())
        q1h_non_kfjp_citations = float(c.reindex(q1h_non_kfjp_cols, fill_value=0).sum())
        q1plus_papers = float(p.reindex(q1plus_cols, fill_value=0).sum())
        q1plus_citations = float(c.reindex(q1plus_cols, fill_value=0).sum())
        kfjp_paper_share = safe_ratio(kfjp_papers, total_papers)
        kfjp_citation_share = safe_ratio(kfjp_citations, total_citations)
        kfjp_cpp = safe_ratio(kfjp_citations, kfjp_papers)
        non_kfjp_cpp = safe_ratio(non_kfjp_citations, non_kfjp_papers)

        row = {
            "subject": subject,
            "scholar": scholar,
            "total_papers": total_papers,
            "total_citations": total_citations,
            "citations_per_paper": safe_ratio(total_citations, total_papers),
            "journal_count": journal_count,
            "known_partition_journal_count": int((p.reindex(known_cols, fill_value=0) > 0).sum()),
            "portfolio_hhi": hhi(p),
            "portfolio_entropy": share_entropy(p),
            "top5_paper_share": topn_share(p, 5),
            "citation_hhi": hhi(c),
            "top5_citation_share": topn_share(c, 5),
            "kfjp_journal_count": int((pk > 0).sum()),
            "non_kfjp_journal_count": int((p.reindex(non_kfjp_cols, fill_value=0) > 0).sum()),
            "q1h_non_kfjp_journal_count": int((p.reindex(q1h_non_kfjp_cols, fill_value=0) > 0).sum()),
            "kfjp_papers": kfjp_papers,
            "kfjp_citations": kfjp_citations,
            "non_kfjp_papers": non_kfjp_papers,
            "non_kfjp_citations": non_kfjp_citations,
            "kfjp_paper_share": kfjp_paper_share,
            "kfjp_citation_share": kfjp_citation_share,
            "kfjp_citation_surplus": kfjp_citation_share - kfjp_paper_share,
            "kfjp_cpp": kfjp_cpp,
            "non_kfjp_cpp": non_kfjp_cpp,
            "kfjp_cpp_log_advantage": np.log1p(kfjp_cpp) - np.log1p(non_kfjp_cpp) if not pd.isna(kfjp_cpp) and not pd.isna(non_kfjp_cpp) else np.nan,
            "kfjp_citation_leverage": safe_ratio(kfjp_citation_share, kfjp_paper_share),
            "q1h_paper_share": safe_ratio(q1h_papers, total_papers),
            "q1h_citation_share": safe_ratio(q1h_citations, total_citations),
            "q1h_non_kfjp_papers": q1h_non_kfjp_papers,
            "q1h_non_kfjp_citations": q1h_non_kfjp_citations,
            "q1h_non_kfjp_paper_share": safe_ratio(q1h_non_kfjp_papers, total_papers),
            "q1h_non_kfjp_citation_share": safe_ratio(q1h_non_kfjp_citations, total_citations),
            "q1plus_paper_share": safe_ratio(q1plus_papers, total_papers),
            "q1plus_citation_share": safe_ratio(q1plus_citations, total_citations),
        }
        row.update(raw_profiles.get(str(scholar), {}))
        rows.append(row)

    return pd.DataFrame(rows)


def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["total_papers", "total_citations", "citations_per_paper", "journal_count", "h_index", "academic_age"]:
        df[f"log_{col}"] = np.log1p(pd.to_numeric(df[col], errors="coerce"))
    for col in ["kfjp_journal_count", "non_kfjp_journal_count", "q1h_non_kfjp_journal_count"]:
        df[f"log_{col}"] = np.log1p(pd.to_numeric(df[col], errors="coerce"))

    # Keep names short and stable for R formulas and plotting labels.
    df["breadth"] = df["log_journal_count"]
    df["concentration"] = df["portfolio_hhi"]
    df["high_tier_pub"] = df["q1h_paper_share"]
    df["kfjp_pub"] = df["kfjp_paper_share"]
    df["kfjp_cite"] = df["kfjp_citation_share"]
    df["kfjp_breadth"] = df["log_kfjp_journal_count"]
    df["non_kfjp_breadth"] = df["log_non_kfjp_journal_count"]
    df["q1h_non_kfjp_breadth"] = df["log_q1h_non_kfjp_journal_count"]
    df["q1h_non_kfjp_pub"] = df["q1h_non_kfjp_paper_share"]
    df["kfjp_surplus"] = df["kfjp_citation_surplus"]
    df["kfjp_cpp_advantage"] = df["kfjp_cpp_log_advantage"]
    df["citation_perf"] = df["log_citations_per_paper"]
    df["citation_volume"] = df["log_total_citations"]
    df["h_perf"] = df["log_h_index"]
    df["productivity"] = df["log_total_papers"]
    df["career_age"] = df["log_academic_age"]
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    partition = load_partition()
    sem_data = pd.concat([build_subject(subject, partition) for subject in SUBJECTS], ignore_index=True)
    sem_data = add_transforms(sem_data)
    sem_data.to_csv(OUT_DIR / "hcr_sem_scholar_variables.csv", index=False, encoding="utf-8-sig")
    sem_data.to_excel(OUT_DIR / "hcr_sem_scholar_variables.xlsx", index=False)

    summary = sem_data.groupby("subject").agg(
        scholars=("scholar", "count"),
        mean_total_papers=("total_papers", "mean"),
        mean_total_citations=("total_citations", "mean"),
        mean_h_index=("h_index", "mean"),
        mean_kfjp_paper_share=("kfjp_paper_share", "mean"),
        mean_kfjp_citation_share=("kfjp_citation_share", "mean"),
        mean_q1h_paper_share=("q1h_paper_share", "mean"),
    )
    summary.to_csv(OUT_DIR / "hcr_sem_variable_summary_by_subject.csv", encoding="utf-8-sig")
    print(summary.round(4).to_string())
    print(f"\nSaved SEM variables to: {OUT_DIR}")


if __name__ == "__main__":
    main()
