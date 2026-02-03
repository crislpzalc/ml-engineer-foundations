"""
Mini-ML Challenge — Noisy and Small Data Scenario

This script simulates a realistic machine learning workflow with
imperfect data (small sample size, noisy features, reduced signal).
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

def load_data():
    """Load the breast cancer dataset and return (X, y, feature_names)."""
    ds = load_breast_cancer()
    return ds.data, ds.target, ds.feature_names

def make_small_dataset(X, y, keep_frac=0.25):
    """
    Subsample the dataset while preserving class proportions.
    This simulates a low-data regime commonly found in real-world problems.
    """
    X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(X, y, train_size=keep_frac, random_state=RANDOM_SEED, stratify=y)

    return X_train_small, y_train_small

def add_noise_features(X, noise_std=0.8):
    """
    Add Gaussian noise to input features to simulate measurement noise.
    Noise is sampled from N(0, noise_std) and applied element-wise.
    """
    noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
    X_noisy = X + noise
    return X_noisy

def drop_features(X, feature_names, drop_frac=0.4):
    """
    Randomly drop a fraction of features to simulate loss of useful signal.
    This mimics scenarios such as missing sensors or unavailable variables.
    """
    n_features = X.shape[1]
    n_drop = int(n_features * drop_frac)
    drop_indices = rng.choice(n_features, size=n_drop, replace=False)

    keep_mask = np.ones(n_features, dtype=bool)
    keep_mask[drop_indices] = False

    X_new = X[:, keep_mask]
    names_new = feature_names[keep_mask]
    return X_new, names_new

def build_bad_dataset(X, y, feature_names):
    """
    Construct a degraded dataset by combining:
    - subsampling (small data),
    - noisy features,
    - random feature removal.

    This simulates a realistic low-quality data scenario.
    """
    X_bad, y_bad = make_small_dataset(X, y, keep_frac=0.25)
    X_bad = add_noise_features(X_bad, noise_std=0.8)
    X_bad, names_bad = drop_features(X_bad, feature_names, drop_frac=0.4)
    return X_bad, y_bad, names_bad

def eval_model(model, X_train, y_train, X_test, y_test, cv=5):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "confusion": confusion_matrix(y_test, y_pred),
    }

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_stats = {"cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std())}
    return test, cv_stats

# Logistic Regression Model Pipeline
def model_logreg(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=10_000, C=C))
    ])

# Random Forest Classifier Model
def model_rf(n_estimators=300, max_depth=None, min_samples_leaf=1):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

# Gradient Boosting Classifier Model
def model_gb():
    return GradientBoostingClassifier(random_state=RANDOM_SEED)

# Logging function for experiment results
def log_try(trials, name, why, test, cv):
    trials.append({
        "name": name,
        "why": why,
        "test_accuracy": test["accuracy"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "cv_mean": cv["cv_mean"],
        "cv_std": cv["cv_std"],
        "confusion": test["confusion"],
    })

def print_summary(trials):
    print("\nExperiment Summary:")
    for t in trials:
        print(f"- {t['name']}: {t['why']}")
        print(f"  Test Accuracy: {t['test_accuracy']:.4f}, Precision: {t['test_precision']:.4f}, Recall: {t['test_recall']:.4f}")
        print(f"  CV Mean Accuracy: {t['cv_mean']:.4f} ± {t['cv_std']:.4f}")
        print(f"  Confusion Matrix:\n{t['confusion']}\n")

def main():
    X, y, feature_names = load_data()
    X_bad, y_bad, names_bad = build_bad_dataset(X, y, feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bad, y_bad, test_size=0.2, random_state=RANDOM_SEED, stratify=y_bad
    )

    trials = []

    # 1) Baseline: Logistic Regression
    m = model_logreg(C=1.0)
    test, cv = eval_model(m, X_train, y_train, X_test, y_test)
    log_try(trials, "logreg_C1", "Baseline simple + scaling + regularization default", test, cv)

    # 2) Improve: more regularization
    m = model_logreg(C=0.2)
    test, cv = eval_model(m, X_train, y_train, X_test, y_test)
    log_try(trials, "logreg_C0.2", "More regularization to reduce overfitting on noisy/small data", test, cv)

    # 3) RF with constrained complexity
    m = model_rf(n_estimators=300, max_depth=8, min_samples_leaf=5)
    test, cv = eval_model(m, X_train, y_train, X_test, y_test)
    log_try(trials, "rf_depth8_leaf5", "Tree ensemble with constrained complexity for robustness", test, cv)

    # 4) GB baseline
    m = model_gb()
    test, cv = eval_model(m, X_train, y_train, X_test, y_test)
    log_try(trials, "grad_boost", "Boosting baseline (can fit complex patterns, risk on noise)", test, cv)

    print_summary(trials)

if __name__ == "__main__":
    main()