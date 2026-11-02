# KFJP Code

Source code for *Scientific impact emerges from complementary journal portfolios*.

## Files

- `kfjp_model.py`: ResCNODE model and shared utilities.
- `train_composition_model.py`: model training.
- `compute_removal_effects.py`: journal-removal analysis and structural criticality.
- `run_eqo_portfolio.py`: EQO portfolio search and size selection.
- `build_hcr_sem_dataset.py`: construction of scholar-level SEM variables.
- `build_kfjp_indicator_test_dataset.py`: KFJP coverage-breadth variables.
- `run_hcr_sem_kfjp_coverage_breadth.R`: piecewise SEM, diagnostics and plots.

## Requirements

Install Python dependencies with `pip install -r requirements.txt`.
The SEM analysis also requires R packages `piecewiseSEM`, `tidyverse`, `ggplot2` and `patchwork` (install separately in R).

## Basic usage

Run from the repository root with your own input files:

```sh
python train_composition_model.py --data-file data.csv --output-dir results_model
python compute_removal_effects.py --input-dir results_model --output-dir results_removal
python run_eqo_portfolio.py --matrix results_removal/removal_effect_matrix.csv --response response.csv --output results_eqo
Rscript run_hcr_sem_kfjp_coverage_breadth.R analysis.csv results_sem
```

The two SEM variable builders can be run in the order listed above. They expect the original Excel input layout under `data/`, or an existing input root specified by `KFJP_SEM_BASE`. The R script accepts an already prepared analysis CSV; without arguments it reads the builders' output under that data root. The academic-age reference year is fixed at 2026.

Input data are not included. SEM estimates describe within-sample associations, not causal effects. Generated plotting layouts need not exactly match the manuscript's final figures.
