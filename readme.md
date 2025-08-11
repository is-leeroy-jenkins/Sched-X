###### Schedule-X
![](https://github.com/is-leeroy-jenkins/Sched-X/blob/master/resources/images/git/schedx.png)
## Schedule-X: Statistical Analysis (CY / BY / PY)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/is-leeroy-jenkins/schedx/blob/master/ipynb/max.ipynb)

A clean, reproducible notebook for exploring **Schedule-X** style budget tables across the **Prior
Year (PY)**, **Current Year (CY)**, and **Budget Year (BY)**. It provides:

- fast **descriptive statistics**,
- **probability distribution** plots,
- **normality diagnostics** (Shapiro–Wilk),
- **confidence intervals**,
- a **one-sample t-test** on CY,
- and tidy **summary tables** you can paste directly into reports.

Perfect for analysts who need a quick statistical pass on budget submissions, fiscal snapshots, and
exploratory analysis of account-level measures.

---

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Data Expectations](#data-expectations)
- [What the Notebook Does](#what-the-notebook-does)
- [Output Artifacts](#output-artifacts)
- [Methods & Statistics](#methods--statistics)
- [Customization Tips](#customization-tips)
- [Repo Structure (suggested)](#repo-structure-suggested)
- [References](#references)
- [License](#license)

---

## Features

- **📊 Descriptive Stats** — `count`, `mean`, `std`, `min/max`, `quartiles`, **skew**, **kurtosis**
  for PY/CY/BY.
- **📈 Distributions** — histograms + KDE for quick shape checks; zeros may be excluded for
  stability.
- **🔎 Normality Testing** — **Shapiro–Wilk** with p-values, per column.
- **🎯 One-Sample t-Test** — tests whether **CY** differs from 0 at α=0.05 (configurable).
- **📏 Confidence Intervals** — 95% CIs (configurable) for key measures.
- **🧹 Sensible Cleaning** — optional zero-filtering and numeric coercion.
- **🧩 Scikit-Learn Helper** — includes a small plotting helper (`plot_decision_regions`) for quick
  ML demos.
- **📝 Copy-able Tables** — neatly formatted DataFrames for pasting into slides or memos.

> Notebook headings you’ll see: **Descriptive Statistics**, **Probability Distributions**, *
*Inferential Statistics**, **Confidence Intervals**, **Normality Testing**, **PY Metrics**, **CY
Metrics**, **BY Metrics**, **t-Test**.

---

## Quickstart

### Option A — Colab (no setup)

Click the badge at the top or open the notebook directly in Colab.  
Upload your CSV (or mount Drive), set the `DATA_PATH`, and run all cells.

### Option B — Local (conda or venv)

```
bash
# 1) create environment
conda create -n schedx python=3.11 -y
conda activate schedx

# 2) install dependencies
pip install pandas numpy scipy matplotlib seaborn scikit-learn mglearn jupyter

# 3) run Jupyter
jupyter notebook
```
## Repo Structure
.
├─ ipynb/
│  └─ max.ipynb
├─ data/
├─ images/
├─ requirements.txt
└─ README.md

## What the Notebook Does

A clear, repeatable pipeline for Schedule-X style analysis across **PY**, **CY**, and **BY**.

### 1) Load & Validate
- Reads your dataset (CSV or DataFrame) and selects the PY/CY/BY columns.
- Coerces them to numeric (safe conversion with errors coerced to `NaN`).
- Optional zero filtering (e.g., drop structural zeros before tests/plots).
- Basic sanity checks (non-null counts, distinct values) so you know what you’re analyzing.

### 2) Descriptive Statistics (PY, CY, BY)
- Computes classic summary stats: `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`.
- Adds **skewness** and **kurtosis** for shape diagnostics.
- Presents **PY Metrics**, **CY Metrics**, **BY Metrics** as compact summary tables for quick copy/paste into memos.

### 3) Probability Distributions
- Plots histograms with KDE overlays for each of PY, CY, BY.
- Helps you spot heavy tails, skew, and multi-modality.
- Optional zero exclusion keeps massed-at-zero distributions from swamping the shapes.

### 4) Normality Testing
- Runs **Shapiro–Wilk** per column; reports statistic and p-value.
- Provides a quick interpretation (e.g., “reject” vs “do not reject” normality at α = 0.05 by default).

### 5) Confidence Intervals
- Computes mean **confidence intervals** for each column (default 95%).
- Uses `t` critical values for modest sample sizes; falls back to normal approximation for large `n`.
- Displays point estimates with lower/upper bounds so you can cite uncertainty, not just point values.

### 6) Inferential Test (One-Sample t-Test on CY)
- Tests **CY** against a configurable baseline (μ₀ = 0 by default).
- Reports **t-statistic**, **degrees of freedom**, **p-value**, and a concise interpretation.
- Encourages policy-relevant μ₀ (e.g., enacted/planned level) when zero is not meaningful.

### 7) (Optional) ML Demo Utility
- Includes a compact `plot_decision_regions` helper (scikit-learn compatible) for quick classification demos.
- Useful when you want to add a pedagogical ML visual without leaving the notebook.

---

### Configuration Knobs (up top in the notebook)
- `DATA_PATH` — file path if loading from CSV.
- `COL_PY`, `COL_CY`, `COL_BY` — rename to match your dataset.
- `DROP_ZEROS` — `True/False` to exclude zeros for plots/tests.
- `ALPHA` — significance level for tests (default `0.05`).
- `CI_LEVEL` — confidence level for CIs (default `0.95`).

### Outputs You Can Reuse
- **Summary tables** for PY/CY/BY (paste into slides or memos).
- **Normality results** with p-values (document your assumptions).
- **Confidence-interval table** (report estimate uncertainty).
- **Distribution plots** (save as images for briefings).
- **t-test readout** (decision & effect direction).

> **Assumptions & Notes**
> - Observations are independent; columns represent the same conceptual measure across PY/CY/BY.
> - If normality is rejected and `n` is small, consider transforming data or using non-parametric tests.
> - Zero handling is configurable because structural zeros can distort both plots and tests.


## Data Expectations

The notebook is designed to work with a **Schedule-X style** dataset containing numeric columns for:

- **PY** — Prior Year  
- **CY** — Current Year  
- **BY** — Budget Year  

At a minimum, the table should look like:

| agency | bureau | account | PY      | CY      | BY      |
|-------:|:------:|:-------:|--------:|--------:|--------:|
| 001    | 10     | 1234    | 1050.25 | 1101.00 | 1149.90 |
| 001    | 20     | 5678    |  450.00 |  465.75 |  480.50 |

### Notes:
- **Column names are configurable** — change them in the configuration block at the start of the notebook.
- **Zero handling** — an option is provided to drop zero values before statistical analysis to avoid skewing distributions or test results.
- **Numeric coercion** — the notebook will attempt to convert your specified columns to numeric types automatically.

Example configuration:

```python
# ---- Configuration ----
DATA_PATH  = "your_data.csv"  # Path to CSV file
COL_PY     = "PY"
COL_CY     = "CY"
COL_BY     = "BY"
DROP_ZEROS = True             # Whether to drop zeros in analysis
ALPHA      = 0.05              # Significance level for statistical tests
CI_LEVEL   = 0.95              # Confidence interval level
