# RéseauFlux — ML-Guided Metabolic Engineering Pipeline

A machine learning pipeline for genome-scale ranking of metabolic interventions, demonstrated on aerobic succinate production in *Escherichia coli* (iJO1366).

> **Author:** Daxlia  
> **Repository:** [github.com/Daxlia/RéseauFlux](https://github.com/Daxlia/ReseauFlux)  
> **License:** Modified MIT with Citation Requirement (see `LICENSE`)

---

## Overview

RéseauFlux integrates constraint-based metabolic modelling (COBRApy) with gradient boosting regression (GBR) to rank all gene knockouts, overexpressions, and downregulations across a genome-scale model — without running FBA for every candidate. The pipeline:

1. Extracts biologically meaningful features per reaction (FVA range, betweenness centrality, shortest paths to target and biomass, subsystem one-hot encoding, flux entropy, etc.)
2. Trains a GBR on rank-normalized succinate flux labels from ~300 enumerated interventions
3. Predicts ranks for the remaining ~1,950 unenumerated reactions
4. Validates top candidates with full two-step FBA (maximize growth → fix growth → maximize target)
5. Extends to double interventions, evolutionary search, Monte Carlo robustness, and multi-target generalization

**Key result:** ATPS4rpp KO is ranked #1 by the model (16.38 mmol/gDW/h succinate, +100× WT), consistent with experimental literature.

---

## Requirements

```bash
pip install cobra highspy scikit-learn networkx numpy pandas matplotlib scipy
pip install straindesign   # optional — enables OptKnock/StrainDesign comparison
```

Python ≥ 3.9. The iJO1366 model is downloaded automatically via `cobra.io.load_model("iJO1366")`.

**Tested with:**
- COBRApy 0.29+
- scikit-learn 1.4+
- HiGHS solver (via `highspy`)

---

## Usage

```bash
python metabolic_ml_pipeline.py
```

All outputs (CSV tables + summary plot) are written to the current directory and packaged into a timestamped ZIP archive at the end of the run.

### Configuration (top of script)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_REACTION` | `"EX_succ_e"` | Exchange reaction to maximize |
| `MAX_ENUM_REACTIONS` | `300` | Interventions to FBA-enumerate for training |
| `GROWTH_FRACTION` | `0.10` | Minimum growth fraction in two-step optimizer |
| `N_TOP` | `10` | Top singles validated with FBA |
| `N_VALIDATE_DOUBLES` | `25` | Top ML-predicted doubles validated with FBA |
| `SEED` | `42` | Random seed |

---

## Output Files

| File | Contents |
|------|----------|
| `ranking_all_interventions.csv` | Full ranked list of all (reaction, kind) pairs |
| `top_validated_strategies.csv` | Top singles with FBA-validated flux and growth |
| `combo_search.csv` | Top ML-predicted double interventions |
| `evolutionary_search.csv` | Triple KO candidates from evolutionary search |
| `monte_carlo_robustness.csv` | Robustness scores under ±20% parameter noise |
| `flux_control_coefficients.csv` | FCC analysis (production bottleneck identification) |
| `gene_knockouts.csv` | Gene-level knockout predictions |
| `ablation_study.csv` | GBR vs pairwise ranker vs ensemble comparison |
| `generalization_holdout.csv` | Holdout test results (70/30 reaction-group split) |
| `literature_comparison.csv` | ML predictions vs known experimental results |
| `multi_target_summary.csv` | Pipeline performance on L-malate and acetate targets |
| `pareto_front.csv` | Pareto-optimal strategies (flux vs growth) |
| `results_summary.png` | 12-panel summary figure |

---

## Results Summary

| Metric | Value |
|--------|-------|
| GBR cross-validation Spearman ρ | 0.557 |
| Holdout Spearman ρ (unseen reaction groups) | 0.602 |
| Top-K precision (CV) | 0.472 |
| Best single intervention | ATPS4rpp KO → 16.38 mmol/gDW/h |
| Best double intervention (ML-guided) | ATPM KO + O2tex KO → 16.80 mmol/gDW/h |
| Literature rank recall (top 10) | 2/3 known KOs ranked in top 10 |
| Multi-target (L-malate) ρ | 0.445 |
| Multi-target (acetate) ρ | 0.491 |

---

## Pipeline Architecture

```
iJO1366 model
    │
    ├─ [1] FVA + network analysis → feature matrix (18 features per reaction)
    ├─ [2] Two-step FBA enumeration → rank-normalized labels (y_rank)
    ├─ [3] GBR training (5-fold GroupKFold CV, grouped by reaction)
    ├─ [4] Rank prediction for all 2,583 reactions × 3 intervention types
    ├─ [5] FBA validation of top-10 singles
    ├─ [6] ML double-intervention prediction + FBA validation of top-25
    ├─ [7] Monte Carlo robustness, FCC, evolutionary search
    └─ [Pub] Holdout generalization, ablation, literature comparison, multi-target
```

**Ablation study results:**

| Model | CV Spearman ρ |
|-------|--------------|
| GBR-only | **0.557** |
| Ensemble (GBR + pairwise) | 0.358 |
| Pairwise-only | 0.159 |

The pairwise ranker was removed from the final pipeline after ablation showed it degrades performance.

---

## Citation

If you use this code or results, please cite:

> Daxlia. *RéseauFlux: Machine Learning-Guided Ranking of Metabolic Interventions for Succinate Overproduction in Escherichia coli.* bioRxiv (2025). DOI: [to be assigned upon submission]

---

## License

Modified MIT with Citation Requirement. See [LICENSE](LICENSE).
