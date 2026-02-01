from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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

# MODEL 1: Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg_model = LogisticRegression(max_iter=10000)
log_reg_metrics = train_and_evaluate(log_reg_model, X_train, X_test, y_train, y_test)
print("Logistic Regression Metrics:", log_reg_metrics)

# MODEL 2: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_metrics = train_and_evaluate(rf_model, X_train, X_test, y_train, y_test)
print("Random Forest Classifier Metrics:", rf_metrics)

# MODEL 3: Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_metrics = train_and_evaluate(gb_model, X_train, X_test, y_train, y_test)
print("Gradient Boosting Classifier Metrics:", gb_metrics)

results = {
    "Logistic Regression": log_reg_metrics,
    "Random Forest Classifier": rf_metrics,
    "Gradient Boosting Classifier": gb_metrics
}

for name, metric in results.items():
    print(f'\===== {name} ===')
    print("accuracy :", metric["accuracy"])
    print("precision:", metric["precision"])
    print("recall   :", metric["recall"])
    print("confusion matrix:\n", metric["confusion_matrix"])

# implementing cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
print("CV mean accuracy:", cv_mean)
print("CV std:", cv_std)
print("Test accuracy:", rf_metrics["accuracy"])
