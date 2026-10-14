# 📊 Interactive ML Visualizer — 23CSE301

A Streamlit web app covering **22 ML algorithms** across Regression, Classification and Clustering, built for the B.Tech CSE capstone project.

---

## 🚀 How to run locally

```bash
# 1. Install the only missing package
pip install streamlit==1.38.0

# 2. Train all models (code ported from ML-capstone/notebooks, run once)
python train_models.py

# 3. Launch the app
streamlit run app/app.py
```

---

## 📦 Project structure

```
ml/
├── train_models.py          # trains all 22 models using the ML-capstone notebook code
├── ML-capstone/             # source notebooks + data (data/hour.csv, data/marketing_campaign.csv)
├── requirements.txt
├── hour.csv                 # regression dataset (bike rentals)
├── marketing_campaign.csv   # classification + clustering dataset
├── models/                  # saved joblib pipelines (auto-created)
├── data/                    # saved test splits, metrics, PCA (auto-created)
└── app/
    ├── app.py               # Home / EDA / results summary
    └── pages/
        ├── 1_Regression.py      # Predict · Results · Learn (10 algorithms)
        ├── 2_Classification.py  # Predict · Results · Learn (10 algorithms)
        ├── 3_Clustering.py      # Results · Learn (2 algorithms)
        └── 4_Playground.py      # Live hyperparameter tuning
    └── utils/
        ├── loaders.py
        ├── toy_data.py
        ├── ui.py
        ├── explainers_regression.py
        ├── explainers_classification.py
        └── explainers_clustering.py
```

---

## 📄 Page-by-page description

| Page | What it does |
|---|---|
| **Home** | Problem statement, dataset summaries, 3 EDA charts, full results tables |
| **Regression** | Predict bike rentals · compare 10 models (R², RMSE, MAE) · residual plot · feature importance |
| **Classification** | Predict campaign response · compare 10 models · confusion matrix · ROC curve |
| **Clustering** | K-Means & Agglomerative results · PCA scatter · elbow curve · dendrogram |
| **Playground** | Retrain KNN / Decision Tree / SVM / K-Means on toy data with live hyperparameter sliders |

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set **Main file path** to `app/app.py`.
4. **Important**: `train_models.py` must be run in CI or locally before deploy, and `models/` + `data/` must be committed to the repo (or use Git LFS).

---

## 🤖 AI assistance

Parts of this application were generated and iteratively refined with AI coding assistance (Antigravity / Gemini). All model training code, dataset preprocessing, and evaluation logic were verified by the project team.
