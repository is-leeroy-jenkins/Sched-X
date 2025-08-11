###### Schedule-X
![](https://github.com/is-leeroy-jenkins/Sched-X/blob/master/resources/images/git/schedx.png)
___
#### A machine-learning pipeline for Combined Schedule (X) reporting.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/is-leeroy-jenkins/schedx/blob/master/schedule-x.ipynb)

## 📊 Features

- **Descriptive Statistics** — `count`, `mean`, `std`, `min/max`, quartiles, **skew**, **kurtosis**.
- **Distributions** — histograms + KDE for PY/CY/BY (optional zero filtering).
- **Normality Testing** — **Shapiro–Wilk** per column with p-values.
- **Confidence Intervals** — mean CIs (95% by default; configurable).
- **Inferential Test** — one-sample **t-test** on CY vs a configurable baseline (default μ₀ = 0).
- **Data Hygiene** — numeric coercion and optional zero exclusion to stabilize analyses.
- **ML Helper** — compact `plot_decision_regions` utility for quick analysis.



## 📈 Table of Contents

- [Data](#-data-expectations)
- [Outputs](#-outputs)
- [Configuration](#-configuration)
- [Requirements](#-requirements)
- [References](#-references)
- [License](#-license)



## 🎯 Quickstart

### Option A — Google Colab (no local setup)

1. Click the **Open In Colab** badge above.
2. Upload your CSV or mount Google Drive.
3. Set `DATA_PATH` near the top of the notebook.
4. **Runtime → Run all**.

### Option B — Local (conda or venv)

```
bash
# 1) Create environment
conda create -n schedx python=3.11 -y
conda activate schedx

# 2) Install dependencies
pip install -U pip wheel setuptools
pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter

# 3) Launch Jupyter
jupyter notebook
```

Open `ipynb/schedule-x.ipynb` and run cells top-to-bottom.

## 📊 Regression

- Linear, Ridge, Lasso, ElasticNet
- Decision Tree, Random Forest, Gradient Boosting
- SVR, KNN, MLP Regressor, Bayesian Ridge, Huber Regressor

## ✅ Classification

- Logistic Regression, Perceptron, SVM, KNN
- Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting
- MLP Classifier, Naive Bayes

## 📊 Diagnostics & Evaluation

- Scatter plots, residuals, precision-recall, ROC curves
- Confusion matrices, ANOVA tests, statistical fitting
- PCA visualizations and correlation heatmaps

## 📁 Data & Engineering

- Excel and CSV ingestion
- Imputation (`SimpleImputer`, `KNNImputer`)
- Scaling (`StandardScaler`, `MinMaxScaler`, `RobustScaler`)
- Feature creation via polynomial expansion
- Dimensionality reduction and outlier detection

## 🏛️ Use in Government

- 📉 Budget Execution forecasting
- 🏛️ OMB MAX A-11 DE 
- 🧮 Audit prep and  anomaly detection

## 🔎 Data Expectations

The notebook is designed for **Schedule-X** style datasets with numeric columns for:

- **PY** — Prior Year
- **CY** — Current Year
- **BY** — Budget Year
- **OY-1** - Out Year 1
- **OY-2** - Out Year 2
- **OY-3** - Out Year 3
- **OY-4** - Out Year 4
- **OY-5** - Out Year 5
- **OY-6** - Out Year 6
- **OY-7** - Out Year 7
- **OY-8** - Out Year 8
- **OY-9** - Out Year 9

A minimal table might look like:

| agency | bureau | account | PY      | CY      | BY      | OY-1      | OY-2       |
|-------:|:------:|:-------:|--------:|--------:|--------:|----------:|-----------:|
| 001    | 10     | 1234    | 1050.25 | 1101.00 | 1149.90 |           |            |
| 001    | 20     | 5678    |  450.00 |  465.75 |  480.50 |           |            |

**Notes**

- Column names are configurable (see [Configuration](#-configuration)).
- The loader coerces specified columns to numeric.
- Optional zero filtering is available to avoid distorting distributions and tests.



## 📏 Outputs

- **Summary Frames** — PY/CY/BY metrics with skew/kurtosis (copy-ready).
- **Distribution Plots** — histograms + KDE overlays per column.
- **Normality Table** — Shapiro–Wilk statistic and p-value with quick interpretation.
- **Confidence Intervals** — mean CIs with lower/upper bounds.
- **t-Test Readout** — t-statistic, degrees of freedom, p-value, and concise summary.

> Pro tip: Right-click plots in Jupyter → “Save image as…” to drop charts directly into briefings.



## 🎯 Configuration

- Set these variables near the top of the notebook:

```
python
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

## 📦 Dependencies

| Package       | Description                          | Link                                               |
|---------------|--------------------------------------|----------------------------------------------------|
| numpy         | Numerical computing                   | [numpy.org](https://numpy.org/)                    |
| pandas        | Data manipulation                     | [pandas.pydata.org](https://pandas.pydata.org/)    |
| matplotlib    | Plotting                              | [matplotlib.org](https://matplotlib.org/)          |
| seaborn       | Statistical plots                     | [seaborn.pydata.org](https://seaborn.pydata.org/)  |
| scikit-learn  | Machine learning models               | [scikit-learn.org](https://scikit-learn.org/)      |
| xgboost       | Extreme gradient boosting             | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| statsmodels   | Statistical modeling & ANOVA          | [statsmodels.org](https://www.statsmodels.org/)    |
| openpyxl      | Excel I/O                             | [openpyxl.readthedocs.io](https://openpyxl.readthedocs.io/) |
| fitz (PyMuPDF)| PDF parsing                           | [pymupdf.readthedocs.io](https://pymupdf.readthedocs.io/) |
| loguru        | Logging                               | [github.com/Delgan/loguru](https://github.com/Delgan/loguru) |




#### Install with:

```
bash
pip install -r requirements.txt
```



## 🧩 References

- **USAspending.gov – Federal Accounts**  
  https://www.usaspending.gov/federal_account
- **OMB Circular A-11 (dataset mirror)**  
  https://www.kaggle.com/datasets/terryeppler/omb-circular-a-11
- **Principles of Federal Appropriations Law (dataset mirror)**  
  https://www.kaggle.com/datasets/terryeppler/principles-of-federal-appropriations-law

> **Disclaimer**: This notebook is for analytical exploration and education.  
> It is **not** an official OMB/Treasury product; validate against authoritative sources before use.

## 📝 License

schedx is published under
the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Sched-X/blob/master/LICENSE.txt).





