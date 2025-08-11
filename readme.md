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

```bash
# 1) create environment
conda create -n schedx python=3.11 -y
conda activate schedx

# 2) install dependencies
pip install pandas numpy scipy matplotlib seaborn scikit-learn mglearn jupyter

# 3) run Jupyter
jupyter notebook
