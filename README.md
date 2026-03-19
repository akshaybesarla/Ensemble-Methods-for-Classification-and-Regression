<div align="center">

<br/>

<h1>🌲 Ensemble Methods in Machine Learning</h1>

<h3>A Comparative Study of Ensemble Methods — Bagging, Random Forests & Gradient Boosting<br/>for Classification and Regression, Built on Decision Tree Base Learners</h3>

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Models%20Built%20from%20Scratch-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Datasets%20%26%20Metrics-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Heatmaps%20%26%20Curves-4C72B0?style=for-the-badge)](https://seaborn.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=for-the-badge)]()

<br/>

> *MSc Data Science and Analytics · Royal Holloway, University of London · September 2024*

<br/>

<img src="./fig5_architecture_diagram.png" alt="Ensemble Architecture Diagram" width="100%"/>

<br/>

**[Overview](#-overview)** &nbsp;·&nbsp; **[Key Features](#-key-features)** &nbsp;·&nbsp; **[Files](#-files-included)** &nbsp;·&nbsp; **[Decision Trees](#-decision-trees-the-base-learner)** &nbsp;·&nbsp; **[Ensemble Methods](#-ensemble-methods)** &nbsp;·&nbsp; **[Results](#-results--visualisations)** &nbsp;·&nbsp; **[Installation](#-installation)**

<br/>

</div>

---

## Overview

This project delivers a rigorous empirical investigation into **ensemble learning** — the practice of combining multiple machine learning models to produce predictions that outperform any individual learner. Submitted as part of an MSc in Data Science and Analytics, it evaluates three foundational ensemble strategies across both classification and regression tasks.

All models — **Decision Tree, Bagging, Random Forest, and Gradient Boosting** — are implemented entirely from scratch in NumPy, with scikit-learn used only for dataset loading, data splitting, and evaluation metrics. This demonstrates a deep, ground-level understanding of every algorithmic component rather than reliance on library black boxes.

Baseline comparisons against **logistic regression** (classification) and **linear regression** (regression) are examined in the full report to contextualise the gains delivered by ensemble approaches.

---

## Key Features

| Feature | Detail |
|---|---|
| **Full from-scratch implementation** | All classifiers and regressors written in NumPy — no sklearn estimator classes |
| **Decision Tree base learner** | Custom CART with Gini impurity (classifier) and MSE / variance reduction (regressor) |
| **Three ensemble strategies** | Bagging, Random Forest, and Gradient Boosting — classifier and regressor variants for each |
| **Systematic hyperparameter tuning** | Grid Search over up to 27 parameter combinations per model |
| **K-Fold Cross-Validation** | k=4 folds for robust, split-independent performance estimation |
| **Learning curve analysis** | Train vs. validation performance tracked across training set sizes (10%–90%) |
| **Heatmap visualisations** | Seaborn heatmaps for multi-parameter Grid Search (Random Forest & Gradient Boosting) |
| **Full metric suite** | Accuracy, Precision, Recall, F1 · MSE, MAE, RMSE, R² |
| **Baseline comparisons** | Logistic Regression & Linear Regression benchmarks discussed in the full report |
| **Academic report** | 12,607-word MSc report with literature review, methodology, and full analysis |

---

## Files Included

| File | Type | Description |
|---|---|---|
| `Ensemble_Methods_Classification.ipynb` | Jupyter Notebook | Decision Tree, Bagging, Random Forest, Gradient Boosting Classifiers on the Iris dataset. Includes Grid Search, K-Fold CV, accuracy plots, and learning curves. |
| `Ensemble_Methods_Regression.ipynb` | Jupyter Notebook | Decision Tree, Bagging, Random Forest, Gradient Boosting Regressors on the Diabetes dataset. Includes Grid Search heatmaps, MSE/MAE plots, RMSE/R² evaluation, and learning curves. |
| `Ensemble_Methods_Report.pdf` | PDF | Full 12,607-word MSc academic report covering background research, methodology, comparative analysis, results, limitations, and recommendations. |
| `fig1_classification_dashboard.png` | PNG | Train/test accuracy, Precision/Recall/F1, and Bagging estimator curve — classification |
| `fig2_regression_dashboard.png` | PNG | MSE, MAE, RMSE, R² comparison across all regression models |
| `fig3_learning_curves.png` | PNG | Learning curves — train vs. validation performance across training sizes |
| `fig4_gridsearch_heatmaps.png` | PNG | Grid Search heatmaps — Random Forest and Gradient Boosting |
| `fig5_architecture_diagram.png` | PNG | Ensemble architecture — Bagging, Random Forest, Gradient Boosting side by side |
| `fig6_decision_tree_depth.png` | PNG | Decision Tree depth vs. performance — bias-variance trade-off |
| `fig7_feature_importance.png` | PNG | Random Forest feature importance — Iris and Diabetes datasets |
| `src/models/` | Python | Modular implementations of all classifiers and regressors |
| `src/evaluation/` | Python | Metrics, cross-validation, and Grid Search functions |
| `results/` | CSV / JSON | Model outputs and metric summaries |
| `requirements.txt` | Text | All dependencies with pinned versions |

---

## Datasets

| | Iris Dataset | Diabetes Dataset |
|---|---|---|
| **Task** | Multi-class Classification | Regression |
| **Samples** | 150 | 442 |
| **Features** | 4 (sepal length, sepal width, petal length, petal width) | 10 (age, sex, BMI, blood pressure, 6 serum values) |
| **Target** | 3 species: Setosa, Versicolor, Virginica | Continuous disease progression score |
| **Split** | 70% train · 15% val · 15% test | 70% train · 15% val · 15% test |
| **Cross-validation** | K-Fold (k=4) | K-Fold (k=4) |

---

## Decision Trees — The Base Learner

Every ensemble method in this project is built on top of a **custom Decision Tree** implemented from scratch. Understanding the base learner is essential — ensemble methods derive all their power from combining many of these trees in different ways.

### How the Decision Tree Works

A Decision Tree recursively partitions the feature space by selecting the feature and threshold that best separates the target values at each node. The tree grows until a stopping criterion is met — maximum depth, minimum samples, or pure nodes — at which point leaf nodes return a prediction.

```
                        [Root Node]
                     Feature ≤ Threshold?
                      /              \
                  [Left]           [Right]
              Feature ≤ t?      Feature ≤ t?
              /        \          /       \
          [Leaf]    [Leaf]    [Leaf]    [Leaf]
        Class=0   Class=1   Class=2   Class=1
```

### Decision Tree Classifier — Gini Impurity

The classifier uses **Gini Impurity** to select the best split at each node.

**Gini Impurity:** &nbsp; `G(t) = 1 − Σ pᵢ²`

**Weighted Gini for split selection:** &nbsp; `Gini_split = (|t_L|/|t|)·G(t_L) + (|t_R|/|t|)·G(t_R)`

```python
def _best_split(self, X, y):
    best_feature, best_threshold = None, None
    best_gini = float('inf')

    for feature in range(X.shape[1]):
        for threshold in np.unique(X[:, feature]):
            left  = X[:, feature] <= threshold
            right = X[:, feature] >  threshold

            gini_left  = 1.0 - sum((np.sum(y[left]  == c) / len(y[left]))  ** 2 for c in np.unique(y))
            gini_right = 1.0 - sum((np.sum(y[right] == c) / len(y[right])) ** 2 for c in np.unique(y))
            gini = (len(y[left]) / len(y)) * gini_left + (len(y[right]) / len(y)) * gini_right

            if gini < best_gini:
                best_gini, best_feature, best_threshold = gini, feature, threshold

    return best_feature, best_threshold
```

### Decision Tree Regressor — Variance Reduction

The regressor uses **MSE and variance reduction** to select the best split.

**MSE at a node:** &nbsp; `MSE(t) = (1/nₜ) Σ(yᵢ − ȳ)²`

**Variance reduction:** &nbsp; `ΔVar = Var(t) − (n_L/nₜ · Var(t_L) + n_R/nₜ · Var(t_R))`

```python
def _build_tree(self, X, y, depth=0):
    best_mse, best_split = float('inf'), None

    for feature_index in range(X.shape[1]):
        for threshold in np.unique(X[:, feature_index]):
            left_mask  = X[:, feature_index] <= threshold
            left_y, right_y = y[left_mask], y[~left_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # Weighted variance — the splitting criterion
            mse = len(left_y) * np.var(left_y) + len(right_y) * np.var(right_y)

            if mse < best_mse:
                best_mse, best_split = mse, (feature_index, threshold)
```

### Decision Tree — Depth vs. Performance

<img src="./fig6_decision_tree_depth.png" alt="Decision Tree Depth vs Performance" width="100%"/>

> *Fig 1 — Decision Tree depth vs. accuracy (Iris) and MSE (Diabetes). The diverging train/validation curves beyond the optimal depth make the bias–variance trade-off directly visible. Overfitting begins at `max_depth ≥ 5` for classification and beyond `max_depth=2` for regression.*

### Why Decision Trees Are Ideal Base Learners

| Property | Why It Matters for Ensembles |
|---|---|
| **High variance** | Ensemble averaging directly targets and neutralises variance |
| **Non-parametric** | Makes no assumptions about data distribution |
| **Handles mixed data** | Continuous and categorical features without preprocessing |
| **Fast to train** | Enables training hundreds of trees within practical time budgets |
| **Naturally non-linear** | Axis-aligned splits approximate complex decision boundaries |
| **Interpretable structure** | Each tree's logic is independently inspectable |

### Why Single Decision Trees Are Not Enough

A single Decision Tree is prone to:
- **Overfitting** — Deep trees memorise training noise, failing to generalise
- **High variance** — Small changes in training data can produce vastly different trees
- **Instability** — Sensitive to outliers and noisy features

These are precisely the weaknesses that **Bagging**, **Random Forests**, and **Gradient Boosting** are designed to address.

---

## Ensemble Methods

<img src="./fig5_architecture_diagram.png" alt="Ensemble Architecture Diagram" width="100%"/>

> *Fig 2 — How each ensemble method combines Decision Trees. Bagging and Random Forest build trees independently in parallel. Gradient Boosting builds sequentially — each new tree corrects the residual errors of all previous trees.*

### Bagging (Bootstrap Aggregating)

Bagging reduces variance by training multiple Decision Trees independently on bootstrap samples of the training data, then aggregating their predictions.

```python
class BaggingClassifier:
    def fit(self, X, y):
        for _ in range(self.n_estimators):
            idx = np.random.choice(len(X), len(X), replace=True)  # Bootstrap sample
            model = DecisionTree()
            model.fit(X[idx], y[idx])
            self.models.append(model)

    def predict(self, X):
        votes = np.array([m.predict(X) for m in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, votes)
```

**What Bagging corrects:** High variance. Each bootstrap sample differs slightly, so each tree is different — averaging them cancels out individual errors without introducing new bias.

---

### Random Forest

Random Forest extends Bagging by introducing an additional layer of randomness: at each node split, only a **random subset of features** is considered (`k = √p` for classification, `k = p/3` for regression). This decorrelates the trees.

```python
class RandomForestClassifier:
    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Random feature subset — the key innovation over Bagging
            features = np.random.choice(X.shape[1], self.max_features, replace=False)
            idx = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTree()
            tree.fit(X[idx][:, features], y[idx])
            self.trees.append((tree, features))
```

**What Random Forest corrects:** Both variance and inter-tree correlation. Random feature selection breaks correlation between trees, making the ensemble more robust.

---

### Gradient Boosting

Unlike Bagging and Random Forest — which build trees in parallel — Gradient Boosting builds trees **sequentially**. Each new tree is trained to correct the **residual errors** of all previous trees.

```python
class GradientBoostingRegressor:
    def fit(self, X, y):
        residual = y.copy()
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)                                # Fit to residuals
            self.models.append(tree)
            residual -= self.learning_rate * tree.predict(X)    # Update residuals

    def predict(self, X):
        return sum(self.learning_rate * m.predict(X) for m in self.models)
```

**What Gradient Boosting corrects:** Bias. Each tree focuses on the mistakes the ensemble is currently making. The learning rate `λ` controls how aggressively each correction is applied.

---

### Method Comparison

| Aspect | Decision Tree | Bagging | Random Forest | Gradient Boosting |
|---|---|---|---|---|
| **Trees built** | 1 | Parallel | Parallel | Sequential |
| **Bootstrap sampling** | No | Yes | Yes | No |
| **Feature randomness** | No | No | Yes (√p / p÷3) | No |
| **Primary fix** | — | Variance | Variance + decorrelation | Bias |
| **Overfitting risk** | High | Low | Low | High (without tuning) |
| **Training speed** | Fast | Moderate | Moderate | Slow |
| **Interpretability** | High | Low | Low | Low |

---

## Hyperparameter Tuning

Grid Search with K-Fold Cross-Validation (k=4) was applied to all models. Full parameter grids:

**Classification (Iris):**

| Model | Parameter | Values Searched | Combinations |
|---|---|---|---|
| Decision Tree | `max_depth` | [2, 3, 4, 5, 6, 7, 8, 9, 10] | 9 |
| Bagging | `n_estimators` | [10, 15, 20, 25, 30] | 5 |
| Random Forest | `n_estimators` × `max_features` | [10, 20, 30] × [2, 3, 4] | 9 |
| Gradient Boosting | `n_estimators` × `learning_rate` × `max_depth` | [10,20,30] × [0.01,0.1,0.2] × [3,5,7] | 27 |

**Regression (Diabetes):**

| Model | Parameter | Values Searched | Combinations |
|---|---|---|---|
| Decision Tree | `max_depth` | [2, 3, 4, 5, 6, 7, 8, 9, 10] | 9 |
| Bagging | `n_estimators` | [10, 15, 20, 25, 30] | 5 |
| Random Forest | `n_estimators` × `max_features` | [10, 20, 30] × [3, 5, 7] | 9 |
| Gradient Boosting | `n_estimators` × `learning_rate` × `max_depth` | [50,100,150] × [0.01,0.1,0.2] × [3,5,7] | 27 |

---

## Results & Visualisations

### Classification — Iris Dataset

| Model | Train Acc | Test Acc | Precision | Recall | F1 Score | Best Parameters |
|---|---|---|---|---|---|---|
| Logistic Regression | — | *(baseline)* | — | — | — | — |
| Decision Tree | 0.99 | 0.96 | 0.95 | 0.95 | 0.95 | `max_depth=6` |
| Bagging | 0.92 | 0.96 | 0.96 | 0.96 | 0.96 | `n_estimators=30` |
| **Random Forest** ⭐ | **0.99** | **1.00** | **1.00** | **1.00** | **1.00** | `n_estimators=10, max_features=2` |
| Gradient Boosting | 0.66 | 0.67 | 0.65 | 0.66 | 0.65 | `n_estimators=30, lr=0.01, max_depth=7` |

<img src="./fig1_classification_dashboard.png" alt="Classification Results Dashboard" width="100%"/>

> *Fig 3 — Left: Train vs Test accuracy per model. Centre: Precision, Recall and F1 breakdown across all models. Right: Bagging validation accuracy as n_estimators increases — plateaus at 30 trees confirming diversity saturation.*

---

### Regression — Diabetes Dataset

| Model | Test MSE | Test MAE | RMSE | R² | Best Parameters |
|---|---|---|---|---|---|
| Linear Regression | — | — | — | *(baseline)* | — |
| Decision Tree | 3829.58 | 50.23 | 61.88 | −3.97 | `max_depth=2` |
| **Bagging** ⭐ | **2651.03** | **41.82** | **51.49** | **0.50** | `n_estimators=25` |
| Random Forest | 3246.13 | 47.25 | 56.97 | 0.38 | `n_estimators=20, max_features=7` |
| Gradient Boosting | 3523.42 | 48.02 | 59.36 | 0.33 | `n_estimators=50, lr=0.1, max_depth=3` |

<img src="./fig2_regression_dashboard.png" alt="Regression Results Dashboard" width="100%"/>

> *Fig 4 — Four-metric comparison: MSE, MAE, RMSE, and R² across all regressors. Bagging leads on all four metrics. Decision Tree R² of −3.97 confirms a single shallow tree performs worse than a simple mean predictor.*

---

### Grid Search Heatmaps

<img src="./fig4_gridsearch_heatmaps.png" alt="Grid Search Heatmaps" width="100%"/>

> *Fig 5 — Validation MSE heatmaps from Grid Search. Left: Random Forest — optimal at `n_estimators=20, max_features=7`. Right: Gradient Boosting — extreme sensitivity to `learning_rate`, with validation MSE ranging from ~4,000 to ~13,000 across configurations.*

---

### Learning Curves

<img src="./fig3_learning_curves.png" alt="Learning Curves" width="100%"/>

> *Fig 6 — Learning curves for all models on both tasks. Solid lines = validation performance, dashed lines = training performance. The large train–validation gap for the Decision Tree confirms high variance. Random Forest converges most consistently.*

**Classification — Key observations:**

| Model | Behaviour | Diagnosis |
|---|---|---|
| Decision Tree | Train accuracy near 100%, validation significantly lower | Overfitting — high variance |
| Bagging | Train/val gap narrows steadily as training size grows | Variance reducing effectively |
| **Random Forest** | Smallest train–val gap across all sizes | Best generalisation |
| Gradient Boosting | Gradual improvement; validation improves with more data | Mild underfitting at low data regimes |

**Regression — Key observations:**

| Model | Behaviour | Diagnosis |
|---|---|---|
| Decision Tree | Low train MSE, high and volatile val MSE | Severe overfitting |
| Bagging | Val MSE decreases steadily with more data | Good variance control |
| **Random Forest** | Smallest train–val MSE gap; most consistent | Best bias–variance balance |
| Gradient Boosting | Lowest val MSE at large training sizes | Strong but data-hungry |

---

### Feature Importance — Random Forest

<img src="./fig7_feature_importance.png" alt="Feature Importance" width="100%"/>

> *Fig 7 — Gini-based feature importance from the best Random Forest configuration. Left: Petal Length and Width dominate Iris classification. Right: BMI and S5 (serum triglycerides) are the strongest predictors of diabetes progression.*

---

### Bias–Variance Summary

| Model | Variance | Bias | Overfitting Risk | Best Use Case |
|---|---|---|---|---|
| Linear / Logistic Regression | Low | High | Low | Linear relationships |
| Decision Tree | Very High | Low | Very High | Interpretable baseline |
| Bagging | Low | Moderate | Low | High-variance base models |
| Random Forest | Low | Low–Moderate | Low | General-purpose; high-dimensional data |
| Gradient Boosting | Moderate | Low | **High** (needs tuning) | Maximum accuracy when tuned |

---

## Key Findings

### Classification — Iris Dataset

- **Random Forest achieved 100% test accuracy** — random feature subset selection decorrelates trees, producing an ensemble diverse enough to perfectly capture the Iris decision boundaries.
- **Bagging matched Decision Tree accuracy (96%)** with a more balanced precision–recall profile, demonstrating effective variance reduction without introducing new bias.
- **Gradient Boosting underperformed significantly** (67%) — the small Iris dataset (150 samples) is insufficient for sequential boosting to leverage its error-correction strategy effectively.
- **Bagging validation accuracy peaked at 30 estimators** — the performance plateau confirms ensemble diversity saturates and additional trees yield diminishing returns.
- **Decision Tree overfitting confirmed at `max_depth ≥ 5`** — training accuracy reached 100% while validation plateaued at 89.5%, a textbook overfit signal.

### Regression — Diabetes Dataset

- **Bagging delivered the best regression performance** (MSE = 2651, R² = 0.50) — a 31% reduction in MSE over the Decision Tree baseline. Averaging 25 bootstrap-trained trees smoothed the high-variance individual predictions.
- **Random Forest (MSE = 3246) trailed Bagging** despite its stronger classification performance. The 10-feature space may be too narrow for random feature selection to deliver meaningful decorrelation benefit.
- **Gradient Boosting required far more data and tuning** — with 442 samples, sequential learners could not converge to a competitive solution without early stopping.
- **The Decision Tree R² of −3.97** confirms predictions worse than a simple mean predictor at `max_depth=2`.
- **Grid Search reduced Gradient Boosting validation MSE by ~70%** (from ~13,000 to ~4,025), underscoring extreme hyperparameter sensitivity.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ensemble-methods-ml.git
cd ensemble-methods-ml

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**

```
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

> **Note:** scikit-learn is used exclusively for `load_iris`, `load_diabetes`, `train_test_split`, `KFold`, and evaluation metrics. Every classifier and regressor is implemented in NumPy from first principles.

---

## Usage

```bash
# Classification experiments — Iris dataset
jupyter notebook Ensemble_Methods_Classification.ipynb

# Regression experiments — Diabetes dataset
jupyter notebook Ensemble_Methods_Regression.ipynb
```

---

## Future Improvements

| Improvement | Rationale |
|---|---|
| **XGBoost / LightGBM / CatBoost benchmarks** | Compare against state-of-the-art optimised libraries |
| **SHAP feature importance analysis** | Explain which features drive predictions in each model |
| **Early stopping for Gradient Boosting** | Prevent overfitting without exhaustive Grid Search |
| **Stacking / blending meta-ensemble** | Combine all model outputs via a meta-learner |
| **MLflow experiment tracking** | Automate logging of all runs, metrics, and artefacts |
| **Streamlit / FastAPI deployment** | Serve the best model as an interactive web application |
| **Parallelisation** | Implement multiprocessing for Bagging and Random Forest |

---

## Acknowledgements

- **Leo Breiman** — foundational work on Bagging (1996) and Random Forests (2001) that underpins this study
- **Jerome H. Friedman** — development of Gradient Boosting Machines (2001)
- **The scikit-learn team** — open access to the Iris and Diabetes benchmark datasets and evaluation utilities
- **UCI Machine Learning Repository** — maintaining accessible benchmark datasets for the research community
- **Royal Holloway, University of London** — academic resources and institutional support throughout the MSc programme

---

## References

- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalisation of online learning and an application to boosting. *JCSS*, 55(1), 119–139.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81–106.
- Scikit-learn Documentation (2023). Ensemble methods. https://scikit-learn.org/stable/modules/ensemble.html

---

## Author

**Candidate Number: 2406856**  
MSc Data Science and Analytics  
Department of Computer Science · Royal Holloway, University of London  
Egham, Surrey TW20 0EX, UK · *Submitted September 2024*

---

<div align="center">

*Built with Python &nbsp;·&nbsp; NumPy &nbsp;·&nbsp; Matplotlib &nbsp;·&nbsp; Seaborn &nbsp;·&nbsp; scikit-learn (datasets & metrics only)*

</div>

