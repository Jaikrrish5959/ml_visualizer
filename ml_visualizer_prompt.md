# Prompt: Build an Interactive ML Visualizer (Streamlit)

## Role
You are an expert Python developer and ML educator. Build a clean, beginner-friendly **Streamlit web app** for a university Machine Learning capstone project (B.Tech CSE, course 23CSE301). The app must (a) explain every algorithm visually in very simple terms and (b) make live predictions using our already-trained models.

---

## 1. Project context
- The project covers three tracks: **Regression (10 algorithms)**, **Classification (10 algorithms)** and **Clustering (2 algorithms)**.
- Models are trained and tuned in Jupyter notebooks and saved with `joblib` into `models/`.
- The app is a bonus deliverable: it must **accept user inputs and return predictions from the trained models**, and it will be **publicly deployed on Streamlit Community Cloud**.
- The app is also used during the presentation and viva, so explanations must be simple, visual and interactive.

**Datasets (fill in before use):**
- Regression dataset: `<name>` | target: `<column>` | features: `<list>`
- Classification dataset: `<name>` | target: `<column>` | classes: `<list>` | features: `<list>`
- Clustering dataset: `<name>` | features: `<list>`

---

## 2. Tech stack (no separate backend)
| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| App + server | Streamlit (multi-page app) |
| ML | scikit-learn (plus XGBoost / LightGBM only if used in notebooks) |
| Data | Pandas, NumPy |
| Interactive charts | Plotly |
| Reused notebook plots | Matplotlib, Seaborn (via `st.pyplot`) |
| Dendrogram | SciPy (`scipy.cluster.hierarchy`) |
| Model loading | joblib |
| Deployment | Streamlit Community Cloud from GitHub |

Do **not** add FastAPI, Flask, a database or authentication.

---

## 3. Folder structure
```
app/
  app.py                     # Overview / home page
  pages/
    1_Regression.py
    2_Classification.py
    3_Clustering.py
    4_Playground.py
  utils/
    loaders.py               # cached model/data loading
    toy_data.py              # cached synthetic datasets
    explainers_regression.py
    explainers_classification.py
    explainers_clustering.py
    ui.py                    # shared card layout helpers
models/                      # *.pkl pipelines (model + scaler + encoder)
data/
requirements.txt             # pinned versions
README.md
```

---

## 4. App layout
- **Sidebar:** page navigation, plus a model selector on each track page.
- **Overview page:** problem statement, dataset summary (shape, missing values, target distribution), 2–3 key EDA plots, and a results summary table.
- **Track pages** have three sections:
  1. **Predict:** an input form built from the real feature columns (`st.number_input`, `st.selectbox`, `st.slider`), a Predict button, and the result shown with `st.success` (class + confidence for classification, value for regression, assigned cluster for clustering).
  2. **Results:** a comparison table of all models (ranked), metric cards via `st.metric`, and the required plots (see section 6).
  3. **Learn the algorithms:** one explainer card per algorithm, chosen with a selectbox (one algorithm shown at a time).
- **Playground page:** hyperparameter sliders that retrain toy models live (k for KNN, max_depth for trees, C for SVM, k for K-Means) and redraw charts and metrics.

---

## 5. Explainer card template (use for every algorithm)
Each card contains, in order:
1. **In simple words:** 1–2 plain-English sentences, no jargon or formulas.
2. **Picture:** an interactive Plotly chart on small toy data (100–200 points, `random_state=42`).
3. **Try it:** one or two controls that change the picture live.
4. **On our data:** that model's real scores, loaded from the saved results.

Use colorblind-friendly colours (e.g. blue `#0072B2` and orange `#E69F00`, or Plotly's `Safe` palette). Every chart needs a title, labelled axes and a legend where applicable.

### 5.1 Regression explainers
Toy data: 1 feature vs. target (e.g. house size vs. price).

| Algorithm | In simple words | Picture and interaction |
|---|---|---|
| Linear Regression | Draws the one straight line closest to all the dots. | Scatter + best-fit line + dotted error lines from each dot. Slope/intercept sliders let the user draw their own line and compare its error to the best fit. |
| Ridge | Straight line with a penalty that stops any feature weight from getting too big. | Bar chart of coefficients (multi-feature toy data). The alpha slider shrinks bars toward zero, never to zero. |
| Lasso | Like Ridge, but can shrink weights to exactly zero, which removes features. | Same bar chart; bars disappear as alpha grows. Shown side by side with Ridge. |
| ElasticNet | A blend of Ridge and Lasso. | Same bar chart with alpha and l1_ratio sliders (0 = Ridge, 1 = Lasso). |
| Polynomial Regression | Bends the line into a curve. | Curved data + fitted curve, degree slider 1–10, train vs. test error shown to reveal overfitting. |
| Decision Tree Regressor | Splits data using yes/no questions and predicts each group's average. | Staircase prediction line over the scatter, with a small tree diagram. max_depth slider. |
| Random Forest Regressor | Many trees on random samples, answers averaged. | Faint staircase lines per tree + one bold average line. n_estimators slider. |
| Gradient Boosting Regressor | Trees built one after another, each fixing earlier mistakes. | "Next round" button; residuals as red bars shrinking per round; error-vs-rounds line. learning_rate slider. |
| SVR | Fits a tube around the line; only dots outside the tube matter. | Line + shaded epsilon tube, outside dots highlighted. Epsilon slider; linear/RBF toggle. |
| KNN Regressor | Predicts the average value of the k closest dots. | Click/slider to place a new x; k nearest dots highlighted with connecting lines; star at the predicted value. k slider. |

### 5.2 Classification explainers
Toy data: two coloured classes in 2D (`make_blobs`, `make_moons`, `make_circles` as needed).

| Algorithm | In simple words | Picture and interaction |
|---|---|---|
| Logistic Regression | Straight dividing line that gives a probability for each class. | Background shaded by probability; user adds a point and sees e.g. "72% orange". Small sigmoid curve beside it. |
| KNN | A new point takes the majority colour of its k nearest neighbours. | Two coloured classes + a grey new point; **arrows from the new point to its k nearest neighbours**; vote tally ("3 orange vs 2 blue → orange"); new point recoloured to the winner. k slider; Euclidean/Manhattan toggle. |
| Naive Bayes (Gaussian) | Picks the most likely class from probabilities, assuming features don't affect each other. | Two bell curves (one per class); slider moves a point along x; live bars show class probabilities. |
| Decision Tree | A flowchart of yes/no questions. | 2D plot split into coloured rectangles + tree diagram; the new point's path is highlighted. max_depth slider. |
| SVM (SVC) | Finds the dividing line with the widest gap between classes. | Boundary + dashed margin lines + circled support vectors. C slider; linear/RBF toggle on circular data. |
| Random Forest | Many trees vote; the majority wins. | Five mini "tree cards" each showing a vote → tally → final answer. Feature-importance bar chart. |
| AdaBoost | Simple rules added one by one; misclassified dots get more attention. | "Next round" button; misclassified dots grow larger; boundary builds up step by step. |
| Gradient Boosting | Each new tree fixes the remaining errors. | Rounds slider showing the boundary sharpening; accuracy-vs-rounds line chart. |
| Bagging | Same tree trained on many random copies of the data, then a vote. | Three bootstrap samples (repeated/missing dots visible) → three boundaries → one smoother combined boundary. |
| MLP (Neural Network) | Layers of connected neurons that learn curved boundaries. | Small network diagram (line thickness = weight) + curved boundary on moons data. Hidden-neurons slider; activation dropdown. |

Extra metrics explainer: an interactive confusion matrix (hover a cell for plain text, e.g. "Predicted Yes, actually No") and a ROC curve with an explanation of AUC.

### 5.3 Clustering explainers
Toy data: unlabeled grey dots.

| Algorithm | In simple words | Picture and interaction |
|---|---|---|
| K-Means | Pick k centres, assign each dot to the nearest centre, move centres to the middle, repeat. | "Assign" / "Move centres" step buttons with centres shown as ✕; elbow curve beside it with the chosen k highlighted. k slider. |
| Agglomerative Hierarchical | Start with every dot alone and keep merging the closest groups. | Dendrogram with a draggable cut-line slider; dots recolour into the resulting clusters. Linkage dropdown (single, complete, average, ward). |

---

## 6. Required real-data visuals (from the project rubric)
- **Regression:** comparison table (R², RMSE, MAE, ranked by R²), residual plot and predicted-vs-actual plot for the best model, feature importance for a tree model.
- **Classification:** a single 10-model table (Accuracy, Precision, Recall, F1-weighted, ROC-AUC with OvR for multi-class), confusion matrix, ROC curves.
- **Clustering:** Silhouette, Davies-Bouldin and Calinski-Harabasz for both algorithms; elbow curve; dendrogram; PCA 2D scatter coloured by cluster for each algorithm; t-SNE plot for at least one.

---

## 7. Technical requirements
- Load each saved **sklearn `Pipeline`** (preprocessing + model) with `joblib`; never refit real models inside the app. Inputs from the form must pass through the same preprocessing used in training.
- Use `@st.cache_resource` for model loading and `@st.cache_data` for datasets and toy data.
- Use `random_state=42` everywhere.
- Toy-model retraining on slider changes must stay under ~1 second.
- Validate form inputs and show friendly error messages.
- Modular code: shared helpers in `utils/`, short functions, docstrings and comments on key steps.
- `requirements.txt` with **pinned versions**; the scikit-learn version must match the one used for training.
- Must run with `streamlit run app/app.py` and deploy unchanged on Streamlit Community Cloud.

---

## 8. Deliverables
1. The complete `app/` folder as specified.
2. `requirements.txt`.
3. A README section covering: how to run locally, how to deploy, a page-by-page description, and a note citing AI assistance (required by the course's academic integrity rules).
4. Build in milestones so each can be its own Git commit:
   1. Skeleton + navigation
   2. Model loading + Predict forms
   3. Regression explainers
   4. Classification explainers
   5. Clustering explainers
   6. Playground
   7. Polish + deploy

## 9. Style
- Language: simple English, short sentences, suitable for second-year students.
- No formulas in "In simple words"; an optional "Show the math" expander is allowed.
- Clean, minimal visuals, consistent colours across all pages.
