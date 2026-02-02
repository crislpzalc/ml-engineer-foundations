from pyexpat import model
from turtle import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import numpy as np


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # training
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluate
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return metrics

# Data Leakage on purpose (for educational purposes only)
def leakage_experiment(model, X, y, test_size=0.2, random_state=42):
    """
    Comparing two different scenarios
        1) Correct way: scaling only training data
        2) Data leakage way: scaling entire dataset before splitting
    """

    # split base
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # correct way
    correct = Pipeline([("scaler", StandardScaler()), ("model", model)])

    correct_metrics = train_and_evaluate(correct, X_train, X_test, y_train, y_test)

    # data leakage way
    model_leaked = clone(model)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)

    leakage_metrics = train_and_evaluate(model_leaked, X_train_leak, X_test_leak, y_train_leak, y_test_leak)

    return correct_metrics, leakage_metrics

# Function to evaluate model with cross-validation
def evaluate_cv(model, X_train, X_test, y_train, y_test, cv=5):
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "cv_mean_accuracy": float(np.mean(scores)),
        "cv_std": float(np.std(scores))
    }

# Try Bias and Variance trade-off with RandomForest 
def bias_variance_experiment(X_train, X_test, y_train, y_test):
    configs ={
        # Underfitting: few trees, shallow depth
        "underfitting": RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42, n_jobs=-1),
        # Overfitting: many trees, deep depth
        "overfitting": RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1),
        # Balanced
        "balanced": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    }
    results = {}
    for name, model in configs.items():
        results[name] = evaluate_cv(model, X_train, X_test, y_train, y_test)
    return results

def main():
    # load dataset
    dataset = load_breast_cancer()
    print(type(dataset))

    X = dataset.data
    y = dataset.target
    print(X.shape, y.shape)
    print(dataset.feature_names)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
 
    # MODEL 1: Logistic Regression
    log_reg_model = LogisticRegression(max_iter=10000)
    log_reg_metrics = train_and_evaluate(log_reg_model, X_train, X_test, y_train, y_test)
    print("Logistic Regression Metrics:", log_reg_metrics)

    # MODEL 2: Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_metrics = train_and_evaluate(rf_model, X_train, X_test, y_train, y_test)
    print("Random Forest Classifier Metrics:", rf_metrics)

    # MODEL 3: Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_metrics = train_and_evaluate(gb_model, X_train, X_test, y_train, y_test)
    print("Gradient Boosting Classifier Metrics:", gb_metrics)

    results = {
        "Logistic Regression": log_reg_metrics,
        "Random Forest Classifier": rf_metrics,
        "Gradient Boosting Classifier": gb_metrics
    }

    for name, metric in results.items():
        print(f'===== {name} ===')
        print("accuracy :", metric["accuracy"])
        print("precision:", metric["precision"])
        print("recall   :", metric["recall"])
        print("confusion matrix:\n", metric["confusion_matrix"])


    # implementing cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print("CV mean accuracy:", cv_mean)
    print("CV std:", cv_std)
    print("Test accuracy:", rf_metrics["accuracy"])


    # Data Leakage Experiment
    log_reg = LogisticRegression(max_iter=10000)
    correct, leaked = leakage_experiment(log_reg, X, y)
    print("CORRECT:", correct["accuracy"], correct["precision"], correct["recall"])
    print("LEAKED :", leaked["accuracy"], leaked["precision"], leaked["recall"])
    print("LEAKED confusion matrix:\n", leaked["confusion_matrix"])


    # Bias-Variance Experiment
    bv_results = bias_variance_experiment(X_train, X_test, y_train, y_test)
    for k, v in bv_results.items():
        print(f"\n== {k} ==")
        print("train_accuracy:", v["train_accuracy"])
        print("test_accuracy:", v["test_accuracy"])
        print("cv_mean_accuracy:", v["cv_mean_accuracy"])
        print("cv_std:", v["cv_std"])

    pass

if __name__ == "__main__":
    main()