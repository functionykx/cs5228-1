# Exploratory Data Analysis (EDA) Report (Train Set Only)

This report documents what the EDA stage does, why each step is performed, and what the outputs look like. The EDA implementation is in [eda.py](file:///Users/bytedance/Documents/trae_projects/cs5228/src/eda.py) and is executed by [run_preprocess_eda.py](file:///Users/bytedance/Documents/trae_projects/cs5228/run_preprocess_eda.py).

## Scope and Data
- Dataset: Telecom churn dataset
- Train split: `churn-bigml-80.csv` (2666 rows)
- Test split: `churn-bigml-20.csv` (667 rows)
- Target: `Churn` (binary)
- EDA scope: **train set only** (no peeking at the test set for analysis/decisions)

## Outputs
All EDA outputs are written to: [artifacts/eda/](file:///Users/bytedance/Documents/trae_projects/cs5228/artifacts/eda)

- `eda_summary.md`: a compact, text-based summary of key statistics and ranked features
- `label_distribution.png`: churn class distribution
- `num_*.png`: numeric feature distributions + churn-grouped boxplots
- `cat_*_freq.png`: categorical frequency plots
- `cat_*_churn_rate.png`: churn rate by category (filtered to categories with at least 20 samples)
- `correlation_heatmap.png`: numeric feature correlation matrix (heatmap)

## Step-by-step Methods (What / Why / How)

### Step 1 — Label distribution overview
**What**
- Compute class counts and the churn rate on the training set.
- Plot a bar chart of class counts (`label_distribution.png`).

**Why**
- Churn datasets are often imbalanced; quantifying this early prevents misleading conclusions later (e.g., overly trusting accuracy).
- The churn rate is also a key “business baseline” to report on the poster.

**How**
- [churn_overview](file:///Users/bytedance/Documents/trae_projects/cs5228/src/eda.py)

### Step 2 — Missing-value scan
**What**
- Count missing values per feature and compute missing-rate.

**Why**
- Missingness impacts preprocessing choices (imputation strategies) and can be predictive itself.
- Even if missingness is low, confirming it prevents surprises during modeling.

**How**
- [missing_value_table](file:///Users/bytedance/Documents/trae_projects/cs5228/src/eda.py)

### Step 3 — Numeric feature summary statistics
**What**
- Produce descriptive statistics (count, mean, std, min, quartiles, max) for numeric features.

**Why**
- Provides quick sanity checks (ranges, extreme values, potential outliers).
- Helps decide whether scaling/normalization is needed for downstream algorithms.

**How**
- [numeric_summary](file:///Users/bytedance/Documents/trae_projects/cs5228/src/eda.py)

### Step 4 — Univariate distributions (numeric + categorical)
**What**
- For each numeric feature:
  - Histogram with KDE (overall distribution)
  - Boxplot grouped by churn label (distribution shift by class)
- For each categorical feature:
  - Frequency bar plot
  - Churn rate bar plot by category (only categories with ≥ 20 samples; top 30 by churn rate)
- Special handling for `State`:
  - Plot top-N states by frequency, aggregate the rest into `Other` to keep plots readable.

**Why**
- Univariate plots are the fastest way to spot:
  - Skewed distributions and outliers (numeric)
  - Features that separate churn vs non-churn (by-class boxplots)
  - Categories associated with high churn rate (categorical churn-rate plot)
- Top-N aggregation avoids overcrowded plots when a feature has high cardinality (e.g., `State`).
- Filtering churn-rate plots to categories with enough samples reduces noisy, unreliable rates from tiny groups.

**How**
- [plot_univariate_distributions](file:///Users/bytedance/Documents/trae_projects/cs5228/src/eda.py)

### Step 5 — Correlation analysis (redundancy / collinearity)
**What**
- Compute the correlation matrix over numeric features and plot it (`correlation_heatmap.png`).
- Extract and list highly correlated feature pairs with `|corr| ≥ 0.95`.

**Why**
- Highly correlated features indicate redundancy and potential collinearity.
  - For linear models, collinearity makes coefficients unstable and interpretation harder.
  - Even for non-linear models, redundancy can inflate feature space without adding information.
- Identifying redundancy early helps select cleaner feature sets for the supervised and clustering stages.

**How**
- [correlation_analysis](file:///Users/bytedance/Documents/trae_projects/cs5228/src/eda.py)

### Step 6 — “Most significant” features via Mutual Information
**What**
- Compute Mutual Information (MI) between each **processed** feature and the churn label.
- Report the top features by MI score.

**Why**
- The project asks to identify the most significant 2–3 features. MI provides a simple, model-agnostic ranking.
- MI can capture non-linear dependencies better than plain correlation with the label.
- Using **processed** features (after One-Hot) makes the ranking compatible with what models will actually see.

**How**
- [top_features_by_mutual_info](file:///Users/bytedance/Documents/trae_projects/cs5228/src/eda.py)

## Results (This Run)
The detailed output is in: [eda_summary.md](file:///Users/bytedance/Documents/trae_projects/cs5228/artifacts/eda/eda_summary.md)

### Class balance
- Training churn rate: **0.1455** (≈ 14.55% churn)

### Missing values
- No missing values were detected in the training features.

### Strongly correlated (redundant) feature pairs
The following pairs show perfect correlation (`corr = 1.0000`), indicating near-duplicate information (likely a fixed-rate transformation between minutes and charge):
- Total day minutes vs Total day charge
- Total eve minutes vs Total eve charge
- Total night minutes vs Total night charge
- Total intl minutes vs Total intl charge

Practical implication:
- For downstream modeling, consider keeping only one from each (minutes or charge), especially for linear models and interpretability.

### Top features by Mutual Information (examples)
The highest MI features in this run include:
- Total day minutes / Total day charge
- International plan (Yes/No)
- Customer service calls
- Number vmail messages

Practical implication:
- These are strong candidates for the “most significant 2–3 features” discussion in the poster and for targeted churn insights.

## Summary for Report (Redundancy / High-Association / One-Hot)

### 1) How redundancy was detected (and what we found)
**Method**
- Compute the Pearson correlation matrix over numeric features on the training set.
- Visualize it as a heatmap (`correlation_heatmap.png`).
- Extract feature pairs with `|corr| ≥ 0.95` as “highly redundant/collinear”.

**Key findings**
- Four (minutes, charge) pairs are perfectly correlated (`corr = 1.0000`):
  - Total day minutes vs Total day charge
  - Total eve minutes vs Total eve charge
  - Total night minutes vs Total night charge
  - Total intl minutes vs Total intl charge

**Interpretation**
- Charges are effectively a fixed-rate linear transformation of minutes.
- For downstream modeling, it is usually sufficient to keep only one from each pair (minutes *or* charge), especially for linear models and interpretability.

### 2) How “strongly associated” features were identified (plots + MI) and results
**Method A — Plots (train only)**
- Numeric features: compare churn vs non-churn via churn-grouped boxplots (`num_*.png`).
- Categorical features: compare churn rate by category (`cat_*_churn_rate.png`, filtered to categories with ≥ 20 samples).

**Method B — Mutual Information (MI) ranking**
- Compute MI between each **processed feature** (after preprocessing/One-Hot) and the churn label.
- Report the top-ranked features in `eda_summary.md`.

**Key findings (examples)**
- International plan
- voice mail plan
- Total day charge
- Customer service calls

- 这里只是相关性分析，不一定存在因果性。前三项可以用价格/服务敏感分析，最后一项expected issue：客服来电次数多-> churn rate高

### 3) Which features were One-Hot encoded (and why)
**One-Hot encoded features**
- `State`
- `Area code` (treated as categorical by design)
- `International plan`
- `Voice mail plan`

**Why**
- These columns represent categories (not continuous quantities).
- One-Hot encoding converts categories into model-friendly numeric columns and avoids imposing an artificial ordering/distance.
