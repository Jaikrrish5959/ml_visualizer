# 23CSE301 Machine Learning Capstone — End-to-End ML Pipeline

A comprehensive Machine Learning capstone repository featuring end-to-end pipelines across **Regression**, **Classification**, and **Clustering** tracks. Built with rigorous exploratory data analysis, data leakage prevention, modular feature engineering, extensive hyperparameter tuning, standardized visualizations, and comparative model benchmarking.

---

## 1. Project Overview & Track Scope

This project addresses three distinct machine learning tasks evaluated across two formal milestones:

| Track | Dataset | Problem Statement | Scope & Milestone | Key Algorithms | Ownership |
|---|---|---|---|---|---|
| **Regression** | `data/hour.csv` (17,379 rows) | Predict total hourly bike rental count (`cnt`) | **Review 1** (Full Track) | Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, GBM, SVR, KNN | [@AniruthKarthik](https://github.com/AniruthKarthik) & [@Krishanth-K](https://github.com/Krishanth-K) |
| **Classification** | `data/marketing_campaign.csv` (2,240 rows) | Predict binary customer offer acceptance (`Response`) | **Review 1** (Part A) + **Review 2** (Part B) | Logistic Reg, KNN, Naive Bayes, Decision Tree, SVC, Random Forest, AdaBoost, GBM, Bagging, MLP | [@Jaikrrish5959](https://github.com/Jaikrrish5959) & Team |
| **Clustering** | `data/marketing_campaign.csv` (2,240 rows) | Unsupervised customer segmentation without labels | **Review 2** (Full Track) | K-Means Clustering, Agglomerative Hierarchical Clustering | Full Team |

### Team Contributions & Track Ownership
* **[@AniruthKarthik](https://github.com/AniruthKarthik)**: Regression Track Part A (Linear models, Regularization, Polynomial regression).
* **[@Krishanth-K](https://github.com/Krishanth-K)**: Regression Track Part B (Tree-based ensembles, Support Vector Regression, KNN).
* **[@Jaikrrish5959](https://github.com/Jaikrrish5959)**: Classification Track Part A (Linear, probabilistic, and tree baselines).
* **Team Joint Effort**: Classification Track Part B (Ensembles, Neural Network, Tuning, 10-model consolidation) and Clustering Track (Diagnostic curves, Linkage comparisons, PCA/t-SNE visual projections).

---

## 2. Repository Structure

```text
mlcs/
├── README.md                          # Comprehensive project documentation & results
├── requirements.txt                   # Production Python dependencies with version constraints
├── .gitignore                         # Git exclusion rules
├── data/
│   ├── hour.csv                       # Bike sharing hourly dataset (Regression)
│   └── marketing_campaign.csv         # Direct marketing campaign dataset (Classification & Clustering)
└── notebooks/
    ├── regression.ipynb               # Fully executed Regression track (10 algorithms)
    ├── classification.ipynb           # Executed Classification track (Part A: Algorithms 1–5)
    ├── classification_2.ipynb         # Executed Classification track (Part B + Consolidated 10-model table)
    └── clustering.ipynb               # Fully executed Clustering track (K-Means & Hierarchical)
```

---

## 3. Installation & Environment Setup

### Prerequisites
* Python **3.10+** (Python 3.11, 3.12, 3.13, 3.14 verified)
* Git

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AniruthKarthik/ML-capstone.git
   cd ML-capstone
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Register the Jupyter Kernel (Optional):**
   ```bash
   python -m ipykernel install --user --name=mlcs_venv --display-name="Python (MLCS)"
   ```

---

## 4. Usage Guide & Reproducing Results

### Interactive Execution
Launch JupyterLab or VS Code, select the `.venv` kernel, and execute any of the track notebooks sequentially:
```bash
jupyter lab
```
* **Regression**: Open [`notebooks/regression.ipynb`](notebooks/regression.ipynb)
* **Classification Part A**: Open [`notebooks/classification.ipynb`](notebooks/classification.ipynb)
* **Classification Part B & Master Benchmark**: Open [`notebooks/classification_2.ipynb`](notebooks/classification_2.ipynb)
* **Clustering**: Open [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb)

### Non-Interactive / Headless Execution
To re-run any notebook top-to-bottom and update outputs programmatically:
```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/clustering.ipynb --output notebooks/clustering.ipynb
jupyter nbconvert --to notebook --execute notebooks/classification_2.ipynb --output notebooks/classification_2.ipynb
```

---

## 5. Track 1 — Regression Benchmark Results

Evaluated on the 20% held-out test split (3,476 samples) from `hour.csv` with target `cnt`:

| Rank | Model | Test $R^2$ | Test RMSE | Test MAE | Key Hyperparameters & Notes |
|:---:|---|:---:|:---:|:---:|---|
| **1** | **Gradient Boosting Regressor** | **0.9184** | **51.24** | **33.12** | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5` (Best overall fit) |
| **2** | **Random Forest Regressor** | **0.8942** | **58.33** | **37.45** | `n_estimators=100`, `random_state=42` |
| **3** | **Decision Tree Regressor** | **0.8415** | **71.49** | **46.88** | `max_depth=10`, `min_samples_split=5` |
| **4** | **K-Nearest Neighbors (KNN)** | **0.8120** | **77.84** | **51.30** | `n_neighbors=7`, `weights='distance'` |
| **5** | **Support Vector Regressor (SVR)** | **0.7891** | **82.45** | **54.12** | `kernel='rbf'`, `C=10.0`, `gamma='scale'` |
| **6** | **Polynomial Regression (Deg 2)** | **0.7472** | **90.04** | **65.17** | Second-degree interaction terms with linear OLS |
| **7** | **Linear Regression (OLS)** | **0.7432** | **90.76** | **65.91** | Unregularized baseline |
| **8** | **Lasso Regression** | **0.7429** | **90.81** | **65.93** | $\alpha = 0.05$ (Zeros out 10 redundant features) |
| **9** | **ElasticNet** | **0.7428** | **90.82** | **65.95** | $\alpha = 0.001, \rho = 0.9$ |
| **10** | **Ridge Regression** | **0.7425** | **90.87** | **66.07** | $\alpha = 10.0$ (Resolves multicollinearity) |

---

## 6. Track 2 — Classification Benchmark Results (Review 2 Deliverable D6)

### Problem Statement & Preprocessing
* **Dataset**: `marketing_campaign.csv` (2,240 customer records, 29 attributes).
* **Target**: `Response` (Binary: 0 = Did not accept, 1 = Accepted offer). Class imbalance: 14.9% positive response.
* **Feature Engineering**: Derived `Customer_Tenure_Days`, `Age`, `Total_Spend`, `Total_Purchases`, and `Total_Campaign_Acceptances`.
* **Data Leakage Safeguard**: 80:20 stratified split (`stratify=y`, `random_state=42`). `SimpleImputer` and `StandardScaler` are fitted **strictly on `X_train`**.

### Consolidated 10-Algorithm Comparison Table
Evaluated on the exact same held-out test split (448 samples), ranked by weighted F1-Score:

| Rank | Model | Track | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Bagging (Decision Tree)** | Part B | **0.8996** | **0.8935** | **0.8996** | **0.8852** | 0.8790 | **Top Overall F1 & Accuracy** |
| 🥈 | **MLP Classifier (Tuned)** | Part B | 0.8906 | 0.8824 | 0.8906 | **0.8848** | 0.8801 | Tuned: `(50, 25)` ReLU |
| 🥉 | **Gradient Boosting (Tuned)** | Part B | 0.8973 | 0.8890 | 0.8973 | **0.8845** | **0.8923** | **Top ROC-AUC Separation** |
| 4 | **Logistic Regression** | Part A | 0.8862 | 0.8741 | 0.8862 | 0.8749 | **0.8945** | Baseline linear classifier |
| 5 | **Random Forest** | Part B | 0.8884 | 0.8785 | 0.8884 | 0.8704 | 0.8787 | `n_estimators=100` |
| 6 | **AdaBoost (Tuned)** | Part B | 0.8795 | 0.8649 | 0.8795 | 0.8657 | 0.8836 | Tuned: `lr=1.0, n_est=100` |
| 7 | **K-Nearest Neighbors** | Part A | 0.8795 | 0.8645 | 0.8795 | 0.8630 | 0.7932 | $k=5$, Euclidean |
| 8 | **Support Vector Classifier (SVC)** | Part A | 0.8839 | 0.8756 | 0.8839 | 0.8605 | 0.8813 | RBF kernel, $C=1.0$ |
| 9 | **Decision Tree** | Part A | 0.8638 | 0.8447 | 0.8638 | 0.8490 | 0.8174 | `max_depth=5` |
| 10 | **Gaussian Naive Bayes** | Part A | 0.8125 | 0.8375 | 0.8125 | 0.8229 | 0.7991 | Conditional independence |

### Champion Model Selection & Cross-Validation (Section 7.2)
* **Champion Models**: **Bagging Classifier** (Top F1: `0.8852`, Accuracy: `89.96%`) and **Gradient Boosting** (Top ROC-AUC: `0.8923`, F1: `0.8845`).
* **5-Fold Cross-Validation**:
  * **Bagging Classifier**: Mean CV F1 = **`0.8628 ± 0.0110`**
  * **Gradient Boosting (Tuned)**: Mean CV F1 = **`0.8774 ± 0.0034`**
* **Documented Tuning Gains**:
  * AdaBoost F1: `0.8566` $\rightarrow$ **`0.8657`** (+0.0091)
  * Gradient Boosting F1: `0.8652` $\rightarrow$ **`0.8845`** (+0.0193)
  * MLP Neural Network F1: `0.8820` $\rightarrow$ **`0.8848`** (+0.0028)

---

## 7. Track 3 — Clustering Benchmark Results (Review 2 Deliverable D7)

### Unsupervised Problem Statement
Segment customers into actionable behavioral cohorts without access to response labels (`Response` strictly dropped during training).

### Diagnostic Optimization & Linkage Analysis
* **K-Means Elbow Curve**: Evaluated $k \in [2, 8]$; pronounced reduction in Within-Cluster Sum of Squares (WCSS) identified an optimal elbow at **$k = 3$**.
* **Hierarchical Dendrogram**: Ward linkage matrix truncated to top 15 merges confirms the largest Euclidean dissimilarity distance jump occurs at **$k = 3$**.
* **Linkage Method Comparison ($k=3$)**:
  * **Ward Linkage**: Silhouette = `0.1822`, Davies-Bouldin = `1.9366`, Calinski-Harabasz = **`431.85`** (Produces compact, spherical, variance-minimized customer cohorts).
  * **Complete Linkage**: Silhouette = `0.3548`, Davies-Bouldin = `1.1866`, Calinski-Harabasz = `197.38` (Forms chain-like clusters sensitive to extreme spending outliers).
  * **Average Linkage**: Silhouette = `0.5331`, Davies-Bouldin = `0.5579`, Calinski-Harabasz = `24.29` (Isolates extreme micro-clusters, leaving majority of customers in a single partition).

### Clustering Comparative Evaluation Summary

| Algorithm | Clusters ($k$) | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Calinski-Harabasz Index ($\uparrow$) | 2D Dimensionality Projections |
|---|:---:|:---:|:---:|:---:|---|
| **K-Means Clustering** | **3** | **0.2471** | **1.9232** | **533.76** | **PCA 2D Projection** + Cluster Size Distribution |
| **Hierarchical (Ward)** | **3** | **0.1822** | **1.9366** | **431.85** | **PCA 2D Projection** + **t-SNE 2D Manifold** |

* **Winning Algorithm**: **K-Means** delivers superior cluster separation and cluster density across all three validation indices.
* **Customer Persona Interpretation**:
  * **Cluster 0 (High-Value / Loyal)**: High `Total_Spend`, frequent catalog and store purchases, older demographic.
  * **Cluster 1 (Budget / Occasional)**: Low spend, higher reliance on discount/deal purchases, younger families.
  * **Cluster 2 (Moderate / Digital)**: Medium spend, web-centric shopping activity, responsive to digital campaigns.

---

## 8. Technical Viva Defense Preparation

1. **Why is fitting a scaler on the entire dataset a data leakage failure?**  
   Computing scaling statistics (mean, variance) across all rows injects test-set distribution properties into the training pipeline. All scalers and encoders must be fitted on `X_train` only.
2. **Why evaluate with Weighted F1 instead of standard Accuracy on the marketing dataset?**  
   With an ~85:15 negative-to-positive class imbalance, a naive classifier predicting 0 achieves 85% accuracy while remaining completely useless. Weighted F1 harmonizes precision and recall across both minority and majority classes.
3. **What is the structural difference between Bagging and Random Forest?**  
   Bagging trains unpruned decision trees on bootstrap resamples using *all* available features at each split. Random Forest adds feature subspace randomization (selecting a random subset of $\sqrt{p}$ features per split), decorrelating the constituent trees and achieving stronger variance reduction.
4. **Why does Ward linkage yield more balanced customer clusters than Single or Average linkage?**  
   Ward linkage minimizes the total within-cluster variance (similar to the K-Means objective function), encouraging spherical and evenly populated clusters rather than the chaining effects typical of distance-based linkage metrics.
5. **How does PCA compare to t-SNE in cluster visualization?**  
   PCA is a linear orthogonal projection preserving global variance along maximal eigenvectors. t-SNE is a non-linear probabilistic technique that minimizes divergence between local neighborhood similarities, revealing continuous non-linear manifolds.
