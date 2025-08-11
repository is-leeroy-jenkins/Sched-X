###### Schedule-X
![](https://github.com/is-leeroy-jenkins/Sched-X/blob/master/resources/images/git/schedx.png)

# Schedule-X 

## 🧩 Background
- At formulation time, agency budget offices key in both policy and baseline estimates—budget
authority, outlays, and receipts—through Schedule X; MAX then auto-populates the related schedules (
A and S) from those entries.
whitehouse.gov

- For the prior-year actuals, MAX doesn’t rely on agency keystrokes: it pulls them from Treasury’s
Governmentwide Treasury Account Symbol Adjusted Trial Balance System (GTAS) after GTAS “lock” in
late October

- This design also keeps the Budget’s “actual” column consistent with the SF-133 Report on Budget
Execution, which agencies submit through GTAS—OMB’s A-11 notes that the Budget’s actuals are derived
from the same data as the SF-133, and Treasury’s guidance emphasizes using the GTAS Period 12
revision window to make GTAS and the Budget match.


- Some schedules have their own external sources/constraints. Employment (Schedule Q) is checked
against OPM’s monthly civilian FTE totals that OPM transmits to OMB

- DoD military employment is provided separately to OMB and reflected in MAX.


- Federal credit programs must compute subsidy rates and financing-account interest with OMB’s Credit
Subsidy Calculator (CSC2); 

- Receipts estimates aren’t free-hand entries either—the Administration’s official receipts forecasts
are produced by Treasury’s Office of Tax Analysis (OTA) and carried through the Budget (and MSR).
U.S. Department of the Treasury

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/is-leeroy-jenkins/BudgetPy/blob/master/ipynb/max.ipynb)

---

## Features

- **📊 Descriptive Statistics** — `count`, `mean`, `std`, `min/max`, quartiles, **skew**, **kurtosis
  **.
- **📈 Distributions** — histograms + KDE for PY/CY/BY (optional zero filtering).
- **🔎 Normality Testing** — **Shapiro–Wilk** per column with p-values.
- **📏 Confidence Intervals** — mean CIs (95% by default; configurable).
- **🎯 Inferential Test** — one-sample **t-test** on CY vs a configurable baseline (default μ₀ = 0).
- **🧹 Data Hygiene** — numeric coercion and optional zero exclusion to stabilize analyses.
- **🧩 ML Helper** — compact `plot_decision_regions` utility for quick scikit-learn demos.
- **📝 Copy-Ready Tables** — concise summary frames for pasting into briefs and slides.

---

## Table of Contents

- [Quickstart](#quickstart)
- [Data Expectations](#data-expectations)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Suggested Repo Structure](#suggested-repo-structure)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)
- [Maintainer](#maintainer)

---

## Quickstart

### Option A — Google Colab (no local setup)

1. Click the **Open In Colab** badge above.
2. Upload your CSV or mount Google Drive.
3. Set `DATA_PATH` near the top of the notebook.
4. **Runtime → Run all**.

### Option B — Local (conda or venv)

```bash
# 1) Create environment
conda create -n schedx python=3.11 -y
conda activate schedx

# 2) Install dependencies
pip install -U pip wheel setuptools
pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter

# 3) Launch Jupyter
jupyter notebook
```

Open `ipynb/max.ipynb` and run cells top-to-bottom.

---

## Data Expectations

The notebook is designed for **Schedule-X** style datasets with numeric columns for:

- **PY** — Prior Year
- **CY** — Current Year
- **BY** — Budget Year

A minimal table might look like:

| agency | bureau | account | PY      | CY      | BY      |
|-------:|:------:|:-------:|--------:|--------:|--------:|
| 001    | 10     | 1234    | 1050.25 | 1101.00 | 1149.90 |
| 001    | 20     | 5678    |  450.00 |  465.75 |  480.50 |

**Notes**

- Column names are configurable (see [Configuration](#configuration)).
- The loader coerces specified columns to numeric.
- Optional zero filtering is available to avoid distorting distributions and tests.

---

## Outputs

- **Summary Frames** — PY/CY/BY metrics with skew/kurtosis (copy-ready).
- **Distribution Plots** — histograms + KDE overlays per column.
- **Normality Table** — Shapiro–Wilk statistic and p-value with quick interpretation.
- **Confidence Intervals** — mean CIs with lower/upper bounds.
- **t-Test Readout** — t-statistic, degrees of freedom, p-value, and concise summary.

> Pro tip: Right-click plots in Jupyter → “Save image as…” to drop charts directly into briefings.

---

## Configuration

Set these variables near the top of the notebook:

```python
# ---- Configuration ----
DATA_PATH  = "your_data.csv"   # Path to CSV
COL_PY     = "PY"
COL_CY     = "CY"
COL_BY     = "BY"
DROP_ZEROS = True              # Exclude zeros for plots/tests
ALPHA      = 0.05              # Significance level
CI_LEVEL   = 0.95              # Confidence interval level
MU_0       = 0.0               # Baseline for one-sample t-test on CY
```

**Tips**

- Use policy-relevant baselines for `MU_0` (e.g., enacted/planned levels) when zero is not
  meaningful.
- Filter the DataFrame by agency/account before running stats to produce slice-specific results.

---

## Suggested Repo Structure

```
.
├─ ipynb/
│  └─ max.ipynb
├─ data/                 # place CSVs here (consider .gitignore for large files)
├─ images/               # optional: exported figures for README
├─ requirements.txt
└─ README.md
```

---

## Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
scikit-learn
jupyter
```

Install with:

```bash
pip install -r requirements.txt
```

---

## References

- **USAspending.gov – Federal Accounts**  
  https://www.usaspending.gov/federal_account
- **OMB Circular A-11 (dataset mirror)**  
  https://www.kaggle.com/datasets/terryeppler/omb-circular-a-11
- **Principles of Federal Appropriations Law (dataset mirror)**  
  https://www.kaggle.com/datasets/terryeppler/principles-of-federal-appropriations-law

> **Disclaimer**: This notebook is for analytical exploration and education.  
> It is **not** an official OMB/Treasury product; validate against authoritative sources before use.

---

## License

**MIT** — Use, adapt, and distribute with attribution.

---

## Maintainer

**Bro** — _“because the code just works.”_
