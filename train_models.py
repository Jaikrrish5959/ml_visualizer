"""
train_models.py
===============
Trains all 22 models (10 regression + 10 classification + 2 clustering) using the
code from the ML-capstone notebooks, and saves them into models/ and data/.

Source notebooks (ML-capstone/notebooks/)
-----------------------------------------
- regression.ipynb        → hour.csv, predict `cnt`
- classification_2.ipynb  → marketing_campaign.csv, predict `Response`
                            (consolidated 10-model comparison, Part A + Part B)
- clustering.ipynb        → marketing_campaign.csv, unsupervised segmentation

Plots/EDA cells are omitted; preprocessing, feature engineering, splits,
hyperparameter selection and final models follow the notebooks.

Every saved model is a full sklearn Pipeline that takes a pandas DataFrame
with the notebook's (engineered) feature columns.

Run:  python train_models.py
"""

import os, json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

# ── Regression models ──────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ── Classification models ──────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, BaggingClassifier,
)
from sklearn.neural_network import MLPClassifier

# ── Clustering models ──────────────────────────────────────────────────────────
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DATA_DIR = os.path.join(ROOT, "ML-capstone", "data")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_STATE = 42


def input_template(X: pd.DataFrame) -> dict:
    """Typical row (median for numeric, mode for categorical) used to fill
    features the app's predict form doesn't ask for."""
    row = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            row[col] = float(X[col].median())
        else:
            row[col] = str(X[col].mode()[0])
    return row


def save_json(obj, filename):
    with open(os.path.join(DATA_DIR, filename), "w") as f:
        json.dump(obj, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# 1. REGRESSION — regression.ipynb
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("REGRESSION (hour.csv -> cnt)  [regression.ipynb]")
print("=" * 60)

bikes = pd.read_csv(os.path.join(SRC_DATA_DIR, "hour.csv"))

bikes_clean = bikes.drop(columns=["instant", "casual", "registered"]).copy()

# Date parsing
bikes_clean["dteday"] = pd.to_datetime(bikes_clean["dteday"])
bikes_clean["day"] = bikes_clean["dteday"].dt.day
bikes_clean["day_of_week"] = bikes_clean["dteday"].dt.dayofweek
bikes_clean = bikes_clean.drop(columns=["dteday"])

# Feature engineering
bikes_clean["rush_hour"] = (
    (bikes_clean["workingday"] == 1) &
    (bikes_clean["hr"].isin([7, 8, 9, 16, 17, 18, 19]))
).astype(int)
bikes_clean["temp_hum_interaction"] = bikes_clean["temp"] * bikes_clean["hum"]
bikes_clean["is_weekend"] = bikes_clean["weekday"].apply(lambda x: 1 if x in [0, 6] else 0)

X = bikes_clean.drop(columns="cnt")
y = bikes_clean["cnt"]

numerical_features = ["temp", "atemp", "hum", "windspeed", "temp_hum_interaction"]
categorical_features = ["season", "weathersit", "mnth", "hr", "weekday"]
binary_features = ["yr", "holiday", "workingday", "is_weekend", "rush_hour", "day"]

strat_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=strat_bins, random_state=RANDOM_STATE
)
print(f"  Training rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
        ("bin", "passthrough", binary_features),
    ],
    remainder="drop",
)
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()


def regression_metrics(actual, predicted):
    return {
        "R2 Score": r2_score(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted),
    }


def evaluate_regression(model, X_train, y_train, X_test, y_test, model_name="Model"):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics = {
        'Model': model_name,
        'Train R2': r2_score(y_train, y_train_pred),
        'Test R2': r2_score(y_test, y_test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred)
    }
    return metrics, model


reg_trained = {}

# --- Linear Regression ---
print("  Training Linear Regression…")
linear_regression_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])
linear_regression_model.fit(X_train, y_train)
reg_trained["Linear Regression"] = linear_regression_model

# --- Ridge (GridSearchCV over alpha) ---
print("  Tuning Ridge…")
ridge_search = GridSearchCV(
    estimator=Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Ridge())]),
    param_grid={"regressor__alpha": np.logspace(-3, 3, 25)},
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=1,
)
ridge_search.fit(X_train, y_train)
reg_trained["Ridge"] = ridge_search.best_estimator_
print(f"    Best Ridge alpha: {ridge_search.best_params_['regressor__alpha']:.4g}")

# --- Lasso (GridSearchCV over alpha) ---
print("  Tuning Lasso…")
lasso_search = GridSearchCV(
    estimator=Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Lasso(max_iter=20_000, random_state=RANDOM_STATE)),
    ]),
    param_grid={"regressor__alpha": np.logspace(-3, 2, 6)},
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=1,
)
lasso_search.fit(X_train, y_train)
reg_trained["Lasso"] = lasso_search.best_estimator_
print(f"    Best Lasso alpha: {lasso_search.best_params_['regressor__alpha']:.4g}")

# --- ElasticNet (GridSearchCV over alpha, l1_ratio) ---
print("  Tuning ElasticNet…")
elastic_net_search = GridSearchCV(
    estimator=Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("regressor", ElasticNet(max_iter=20_000, random_state=RANDOM_STATE)),
    ]),
    param_grid={
        "regressor__alpha": [0.001, 0.01, 0.1, 1.0],
        "regressor__l1_ratio": [0.1, 0.5, 0.9],
    },
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=1,
)
elastic_net_search.fit(X_train, y_train)
reg_trained["ElasticNet"] = elastic_net_search.best_estimator_
print(f"    Best ElasticNet parameters: {elastic_net_search.best_params_}")


# --- Polynomial Regression (degree chosen on a validation split) ---
def make_poly_prep(degree):
    return ColumnTransformer(
        transformers=[
            ("poly_num", Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ]), numerical_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder="drop",
    )


print("  Selecting polynomial degree…")
X_poly_train, X_poly_validation, y_poly_train, y_poly_validation = train_test_split(
    X_train, y_train, test_size=0.20, random_state=RANDOM_STATE,
)
polynomial_degree_rows = []
for degree in [1, 2]:
    candidate_model = Pipeline(steps=[
        ("preprocessor", make_poly_prep(degree)),
        ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    candidate_model.fit(X_poly_train, y_poly_train)
    validation_metrics = regression_metrics(y_poly_validation, candidate_model.predict(X_poly_validation))
    polynomial_degree_rows.append({"Degree": degree, **validation_metrics})
polynomial_degree_results = pd.DataFrame(polynomial_degree_rows).sort_values("RMSE", ignore_index=True)
best_polynomial_degree = int(polynomial_degree_results.loc[0, "Degree"])
print(f"    Selected polynomial degree: {best_polynomial_degree}")

polynomial_model = Pipeline(steps=[
    ("preprocessor", make_poly_prep(best_polynomial_degree)),
    ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
])
polynomial_model.fit(X_train, y_train)
reg_trained["Polynomial Regression"] = polynomial_model

# --- Decision Tree (max_depth sweep) ---
print("  Sweeping Decision Tree max_depth…")
depths = [3, 5, 7, 10, 15, None]
dt_results = []
for d in depths:
    pipe_dt = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE))
    ])
    metrics, _ = evaluate_regression(pipe_dt, X_train, y_train, X_test, y_test, model_name=f"DT (depth={d})")
    dt_results.append(metrics)
df_dt = pd.DataFrame(dt_results)
best_depth = depths[df_dt['Test R2'].idxmax()]
print(f"    Best max_depth: {best_depth}")

_, model_dt = evaluate_regression(
    Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE))
    ]),
    X_train, y_train, X_test, y_test, model_name="Decision Tree",
)
reg_trained["Decision Tree Regressor"] = model_dt

# --- Random Forest (n_estimators x max_depth sweep) ---
print("  Sweeping Random Forest configs…")
rf_configs = [
    (50, 10), (50, 15), (50, None),
    (100, 10), (100, 15), (100, None),
    (200, 10), (200, 15), (200, None),
]
rf_results = []
for n_est, depth in rf_configs:
    pipe_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=n_est, max_depth=depth,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    metrics, _ = evaluate_regression(pipe_rf, X_train, y_train, X_test, y_test, model_name=f"RF (n={n_est}, depth={depth})")
    rf_results.append(metrics)
df_rf = pd.DataFrame(rf_results)
best_rf_config = rf_configs[df_rf['Test R2'].idxmax()]
print(f"    Best RF config: n_estimators={best_rf_config[0]}, max_depth={best_rf_config[1]}")

_, model_rf = evaluate_regression(
    Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=best_rf_config[0], max_depth=best_rf_config[1],
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ]),
    X_train, y_train, X_test, y_test, model_name="Random Forest",
)
reg_trained["Random Forest Regressor"] = model_rf

# --- Gradient Boosting (n_estimators x learning_rate sweep) ---
print("  Sweeping Gradient Boosting configs…")
gb_configs = [
    (100, 0.05), (100, 0.1), (100, 0.2),
    (200, 0.05), (200, 0.1), (200, 0.2),
]
gb_results = []
for n_est, lr in gb_configs:
    pipe_gb = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=n_est, learning_rate=lr,
            max_depth=5, random_state=RANDOM_STATE
        ))
    ])
    metrics, _ = evaluate_regression(pipe_gb, X_train, y_train, X_test, y_test, model_name=f"GB (n={n_est}, lr={lr})")
    gb_results.append(metrics)
df_gb = pd.DataFrame(gb_results)
best_gb_config = gb_configs[df_gb['Test R2'].idxmax()]
print(f"    Best GB config: n_estimators={best_gb_config[0]}, learning_rate={best_gb_config[1]}")

_, model_gb = evaluate_regression(
    Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=best_gb_config[0], learning_rate=best_gb_config[1],
            max_depth=5, random_state=RANDOM_STATE
        ))
    ]),
    X_train, y_train, X_test, y_test, model_name="Gradient Boosting",
)
reg_trained["Gradient Boosting Regressor"] = model_gb

# --- SVR (kernel x C sweep) ---
print("  Sweeping SVR configs (slowest step)…")
svr_configs = [
    ('rbf', 0.1), ('rbf', 1), ('rbf', 10), ('rbf', 100),
    ('linear', 0.1), ('linear', 1), ('linear', 10),
]
svr_results = []
for kernel, c_val in svr_configs:
    pipe_svr = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', SVR(kernel=kernel, C=c_val))
    ])
    metrics, _ = evaluate_regression(pipe_svr, X_train, y_train, X_test, y_test, model_name=f"SVR ({kernel}, C={c_val})")
    svr_results.append(metrics)
    print(f"    SVR ({kernel}, C={c_val}): Test R2={metrics['Test R2']:.4f}")
df_svr = pd.DataFrame(svr_results)
best_svr_config = svr_configs[df_svr['Test R2'].idxmax()]
print(f"    Best SVR config: kernel={best_svr_config[0]}, C={best_svr_config[1]}")

_, model_svr = evaluate_regression(
    Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', SVR(kernel=best_svr_config[0], C=best_svr_config[1]))
    ]),
    X_train, y_train, X_test, y_test, model_name="SVR",
)
reg_trained["SVR"] = model_svr

# --- KNN (k sweep) ---
print("  Sweeping KNN k…")
k_values = [3, 5, 7, 11, 15, 21]
knn_results = []
for k in k_values:
    pipe_knn = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(n_neighbors=k, n_jobs=-1))
    ])
    metrics, _ = evaluate_regression(pipe_knn, X_train, y_train, X_test, y_test, model_name=f"KNN (k={k})")
    knn_results.append(metrics)
df_knn = pd.DataFrame(knn_results)
best_k = k_values[df_knn['Test R2'].idxmax()]
print(f"    Best k: {best_k}")

_, model_knn = evaluate_regression(
    Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(n_neighbors=best_k, n_jobs=-1))
    ]),
    X_train, y_train, X_test, y_test, model_name="KNN",
)
reg_trained["KNN Regressor"] = model_knn

# --- Evaluate + save ---
reg_results = {}
for name, pipe in reg_trained.items():
    m = regression_metrics(y_test, pipe.predict(X_test))
    reg_results[name] = {"R2": round(float(m["R2 Score"]), 4),
                         "RMSE": round(float(m["RMSE"]), 4),
                         "MAE": round(float(m["MAE"]), 4)}
    joblib.dump(pipe, os.path.join(MODELS_DIR, f"reg_{name.replace(' ', '_')}.pkl"), compress=3)
    print(f"  {name:<28} R²={reg_results[name]['R2']:.4f}  RMSE={reg_results[name]['RMSE']:.2f}  MAE={reg_results[name]['MAE']:.2f}")

save_json(reg_results, "reg_results.json")
X_test.to_pickle(os.path.join(DATA_DIR, "X_reg_test.pkl"))
np.save(os.path.join(DATA_DIR, "y_reg_test.npy"), y_test.values.astype(float))
save_json(input_template(X_train), "reg_input_template.json")

# Feature importances from the best tree-based model (as in the notebook)
tree_names = ["Random Forest Regressor", "Gradient Boosting Regressor", "Decision Tree Regressor"]
best_tree_name = max(tree_names, key=lambda n: reg_results[n]["R2"])
importances = reg_trained[best_tree_name].named_steps["regressor"].feature_importances_
pd.DataFrame({"feature": feature_names, "importance": importances}) \
    .sort_values("importance", ascending=False).head(15) \
    .to_csv(os.path.join(DATA_DIR, "reg_feature_importance.csv"), index=False)
save_json({"model": best_tree_name}, "reg_feature_importance_model.json")
print(f"  Feature importances saved ({best_tree_name}).")

# ──────────────────────────────────────────────────────────────────────────────
# 2. CLASSIFICATION — classification_2.ipynb (consolidated 10 models)
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("CLASSIFICATION (marketing_campaign.csv -> Response)  [classification_2.ipynb]")
print("=" * 60)

df = pd.read_csv(os.path.join(SRC_DATA_DIR, "marketing_campaign.csv"), sep=';')
df_c = df.copy()

# 1. Derive Customer Tenure from Dt_Customer
df_c['Dt_Customer'] = pd.to_datetime(df_c['Dt_Customer'], format='%Y-%m-%d')
df_c['Customer_Tenure_Days'] = (pd.Timestamp('2014-07-01') - df_c['Dt_Customer']).dt.days

# 2. Drop non-predictive identifiers and zero-variance constants
df_c.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], inplace=True, errors='ignore')

# 3. Feature Engineering
df_c['Age'] = 2015 - df_c['Year_Birth']
df_c['Total_Spend'] = (
    df_c['MntWines'] + df_c['MntFruits'] + df_c['MntMeatProducts'] +
    df_c['MntFishProducts'] + df_c['MntSweetProducts'] + df_c['MntGoldProds']
)
df_c['Total_Purchases'] = (
    df_c['NumWebPurchases'] + df_c['NumCatalogPurchases'] + df_c['NumStorePurchases']
)
df_c['Total_Campaign_Acceptances'] = (
    df_c['AcceptedCmp1'] + df_c['AcceptedCmp2'] + df_c['AcceptedCmp3'] +
    df_c['AcceptedCmp4'] + df_c['AcceptedCmp5']
)
df_c.drop(columns=['Year_Birth'], inplace=True, errors='ignore')

# 4. Stratified Train / Test Split (80:20)
X = df_c.drop(columns=['Response'])
y = df_c['Response']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# 5. Pipeline Preprocessing (Impute -> Scale -> One-Hot Encode)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
print(f"  X_train shape: {X_train_proc.shape} | X_test shape: {X_test_proc.shape}")

# --- Part B: tuned models (GridSearchCV on the preprocessed training split) ---
print("  Tuning AdaBoost…")
grid_ada = GridSearchCV(
    AdaBoostClassifier(random_state=RANDOM_STATE),
    param_grid={'n_estimators': [30, 50, 100], 'learning_rate': [0.1, 0.5, 1.0]},
    cv=5, scoring='f1_weighted', n_jobs=-1,
)
grid_ada.fit(X_train_proc, y_train)
print(f"    Best Parameters: {grid_ada.best_params_} (CV F1: {grid_ada.best_score_:.4f})")

print("  Tuning Gradient Boosting…")
grid_gbm = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    param_grid={'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]},
    cv=5, scoring='f1_weighted', n_jobs=-1,
)
grid_gbm.fit(X_train_proc, y_train)
print(f"    Best Parameters: {grid_gbm.best_params_} (CV F1: {grid_gbm.best_score_:.4f})")

print("  Tuning MLP…")
grid_mlp = GridSearchCV(
    MLPClassifier(max_iter=300, random_state=RANDOM_STATE),
    param_grid={'hidden_layer_sizes': [(50,), (50, 25), (100,)], 'activation': ['relu', 'tanh']},
    cv=5, scoring='f1_weighted', n_jobs=-1,
)
grid_mlp.fit(X_train_proc, y_train)
print(f"    Best Parameters: {grid_mlp.best_params_} (CV F1: {grid_mlp.best_score_:.4f})")

# Final 10 models as used in the notebook's consolidated comparison table
# (Part A baselines + Random Forest/Bagging baselines + tuned AdaBoost/GBM/MLP)
clf_models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "KNN":                  KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":          GaussianNB(),
    "Decision Tree":        DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "SVM (SVC)":            SVC(probability=True, random_state=RANDOM_STATE),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "AdaBoost":             clone(grid_ada.best_estimator_),
    "Gradient Boosting":    clone(grid_gbm.best_estimator_),
    "Bagging":              BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                                              n_estimators=50, random_state=RANDOM_STATE),
    "MLP (Neural Network)": clone(grid_mlp.best_estimator_),
}

clf_results = {}
for name, estimator in clf_models.items():
    # Wrap preprocessor + model in one pipeline so the saved model takes raw DataFrames.
    pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    yprob = pipe.predict_proba(X_test)[:, 1]

    clf_results[name] = {
        "Accuracy":  round(float(accuracy_score(y_test, yp)), 4),
        "Precision": round(float(precision_score(y_test, yp, average='weighted', zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_test, yp, average='weighted', zero_division=0)), 4),
        "F1":        round(float(f1_score(y_test, yp, average='weighted')), 4),
        "ROC_AUC":   round(float(roc_auc_score(y_test, yprob)), 4),
    }
    fname = f"clf_{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    joblib.dump(pipe, os.path.join(MODELS_DIR, fname))
    r = clf_results[name]
    print(f"  {name:<22} Acc={r['Accuracy']:.4f}  wF1={r['F1']:.4f}  AUC={r['ROC_AUC']:.4f}")

save_json(clf_results, "clf_results.json")
X_test.to_pickle(os.path.join(DATA_DIR, "X_clf_test.pkl"))
np.save(os.path.join(DATA_DIR, "y_clf_test.npy"), y_test.values)
save_json(input_template(X_train), "clf_input_template.json")
save_json({c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols}, "clf_categories.json")

# ──────────────────────────────────────────────────────────────────────────────
# 3. CLUSTERING — clustering.ipynb
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("CLUSTERING (marketing_campaign.csv)  [clustering.ipynb]")
print("=" * 60)

df_raw = pd.read_csv(os.path.join(SRC_DATA_DIR, "marketing_campaign.csv"), sep=';')

# Drop non-informative IDs and zero-variance constants
df_clean = df_raw.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], errors='ignore')

# Feature Engineering
df_clean['Age'] = 2015 - df_clean['Year_Birth']
df_clean['Total_Spend'] = (
    df_clean['MntWines'] + df_clean['MntFruits'] + df_clean['MntMeatProducts'] +
    df_clean['MntFishProducts'] + df_clean['MntSweetProducts'] + df_clean['MntGoldProds']
)
df_clean['Total_Purchases'] = (
    df_clean['NumWebPurchases'] + df_clean['NumCatalogPurchases'] + df_clean['NumStorePurchases']
)
df_clean['Total_Campaign_Acceptances'] = (
    df_clean['AcceptedCmp1'] + df_clean['AcceptedCmp2'] + df_clean['AcceptedCmp3'] +
    df_clean['AcceptedCmp4'] + df_clean['AcceptedCmp5']
)
df_clean.drop(columns=['Year_Birth'], inplace=True, errors='ignore')

# Drop target column for unsupervised clustering
X_features = df_clean.drop(columns=['Response'], errors='ignore')

num_cols = X_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_features.select_dtypes(include=['object', 'string']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), cat_cols),
])
X_scaled = preprocessor.fit_transform(X_features)
print(f"  Preprocessed feature matrix shape: {X_scaled.shape}")

# 2D PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Elbow curve & silhouette, k = 2..8
k_range = list(range(2, 9))
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    labels_k = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels_k))

# Final K-Means
optimal_k = 3
kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', n_init='auto', random_state=42)
kmeans_labels = kmeans_model.fit_predict(X_scaled)
sil_km = silhouette_score(X_scaled, kmeans_labels)
db_km = davies_bouldin_score(X_scaled, kmeans_labels)
ch_km = calinski_harabasz_score(X_scaled, kmeans_labels)
print(f"  K-Means (k={optimal_k}): Sil={sil_km:.4f}  DB={db_km:.4f}  CH={ch_km:.2f}")

# Final Agglomerative (Ward)
agg_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg_model.fit_predict(X_scaled)
sil_agg = silhouette_score(X_scaled, agg_labels)
db_agg = davies_bouldin_score(X_scaled, agg_labels)
ch_agg = calinski_harabasz_score(X_scaled, agg_labels)
print(f"  Agglomerative (Ward, k=3): Sil={sil_agg:.4f}  DB={db_agg:.4f}  CH={ch_agg:.2f}")

joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", kmeans_model)]),
            os.path.join(MODELS_DIR, "clu_K-Means.pkl"))
joblib.dump((agg_model, preprocessor, agg_labels), os.path.join(MODELS_DIR, "clu_Agglomerative.pkl"))

save_json({
    "K-Means": {"Silhouette": round(float(sil_km), 4), "Davies_Bouldin": round(float(db_km), 4),
                "Calinski_Harabasz": round(float(ch_km), 4)},
    "Agglomerative Hierarchical": {"Silhouette": round(float(sil_agg), 4), "Davies_Bouldin": round(float(db_agg), 4),
                                   "Calinski_Harabasz": round(float(ch_agg), 4)},
}, "clu_results.json")
np.save(os.path.join(DATA_DIR, "clu_X_pca.npy"),      X_pca)
np.save(os.path.join(DATA_DIR, "clu_km_labels.npy"),  kmeans_labels)
np.save(os.path.join(DATA_DIR, "clu_agg_labels.npy"), agg_labels)
np.save(os.path.join(DATA_DIR, "clu_X_scaled.npy"),   X_scaled)
save_json({"k": k_range, "inertia": [float(v) for v in inertias],
           "silhouette": [float(v) for v in sil_scores]}, "clu_elbow.json")
print("  PCA 2D, labels and elbow curve saved.")

print()
print("[OK] All models trained and saved to models/")
print("[OK] All result files saved to data/")
print()
print("Run the app with:  streamlit run app/app.py")
