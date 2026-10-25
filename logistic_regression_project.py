from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

X = df[["mean radius", "mean texture"]]

y = (data.target == 0).astype(int)
print("Selected features:", X.columns.tolist())
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nMissing values:")
print(X.isnull().sum())

print("\nClass counts:")
print(pd.Series(y).value_counts().sort_index())
indices = np.arange(len(y))

dev_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    stratify=y,
    random_state=42
)

train_idx, val_idx = train_test_split(
    dev_idx,
    test_size=0.25,
    stratify=y[dev_idx],
    random_state=42
)
X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
X_test = X.iloc[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]
print("\nSplit sizes:")
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))
print("\nOverlap checks:")
print("Train vs Validation:", len(set(train_idx) & set(val_idx)))
print("Train vs Test:", len(set(train_idx) & set(test_idx)))
print("Validation vs Test:", len(set(val_idx) & set(test_idx)))
print("\nClass proportions:")
print("Train:", pd.Series(y_train).value_counts(normalize=True).sort_index())
print("Validation:", pd.Series(y_val).value_counts(normalize=True).sort_index())
print("Test:", pd.Series(y_test).value_counts(normalize=True).sort_index())
print("\nClass counts by split:")
print("Train:", pd.Series(y_train).value_counts().sort_index().to_dict())
print("Validation:", pd.Series(y_val).value_counts().sort_index().to_dict())
print("Test:", pd.Series(y_test).value_counts().sort_index().to_dict())

all_split_indices = set(train_idx) | set(val_idx) | set(test_idx)
print("All rows covered:", len(all_split_indices) == len(indices))
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("\nScaled training mean:", X_train_scaled.mean(axis=0))
print("Scaled training std:", X_train_scaled.std(axis=0))
X_train_aug = np.column_stack([
    np.ones(len(X_train_scaled)),
    X_train_scaled
])

beta = np.zeros(X_train_aug.shape[1])

learning_rate = 0.1
epochs = 20000
loss_history = []
print("\nX_train_aug shape:", X_train_aug.shape)
print("Initial beta:", beta)
def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))
print("\nSigmoid test at z=0:", sigmoid(0))
def binary_cross_entropy_loss(z, y_true):
    return np.mean(np.logaddexp(0, z) - y_true * z)
initial_z = X_train_aug @ beta
initial_loss = binary_cross_entropy_loss(initial_z, y_train)

loss_history.append(initial_loss)

print("\nInitial loss:", initial_loss)
for epoch in range(epochs):
    z = X_train_aug @ beta
    p = sigmoid(z)

    gradient = X_train_aug.T @ (p - y_train) / len(y_train)

    beta = beta - learning_rate * gradient

    z_updated = X_train_aug @ beta
    loss = binary_cross_entropy_loss(z_updated, y_train)
    loss_history.append(loss)
print("Final loss:", loss_history[-1])
print("Learned beta:", beta)
epochs_axis = np.arange(len(loss_history))

plt.figure()
plt.plot(epochs_axis, loss_history)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("NumPy Logistic Regression Training Loss")

plt.tight_layout()
plt.savefig(REPORTS / "training_loss.png", dpi=150)
plt.close()

print("Saved training loss plot to:", REPORTS / "training_loss.png")
sk_model = LogisticRegression(
    C=np.inf,
    solver="lbfgs",
    max_iter=10000,
    tol=1e-9
)

sk_model.fit(X_train_scaled, y_train)

sk_beta = np.concatenate([
    sk_model.intercept_,
    sk_model.coef_[0]
])

print("\nNumPy beta:", beta)
print("sklearn beta:", sk_beta)
print("Coefficient difference:", beta - sk_beta)
X_val_aug = np.column_stack([
    np.ones(len(X_val_scaled)),
    X_val_scaled
])

numpy_val_prob = sigmoid(X_val_aug @ beta)

sklearn_val_prob = sk_model.predict_proba(X_val_scaled)[:, 1]

mean_abs_prob_diff = np.mean(
    np.abs(numpy_val_prob - sklearn_val_prob)
)

print("\nMean absolute validation probability difference:",
      mean_abs_prob_diff)
baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train_scaled, y_train)

baseline_val_prob = baseline.predict_proba(X_val_scaled)[:, 1]

print("\nBaseline class probabilities:", baseline.class_prior_)
def evaluate_model(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    loss = log_loss(y_true, probabilities)

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": loss
    }
baseline_val_results = evaluate_model(
    y_val,
    baseline_val_prob,
    0.50
)

print("\nBaseline validation results:")
print(baseline_val_results)
thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

threshold_rows = []

for threshold in thresholds:
    results = evaluate_model(
        y_val,
        numpy_val_prob,
        threshold
    )

    threshold_rows.append({
        "threshold": threshold,
        "tn": results["tn"],
        "fp": results["fp"],
        "fn": results["fn"],
        "tp": results["tp"],
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"]
    })

threshold_results = pd.DataFrame(threshold_rows)

print("\nThreshold results:")
print(threshold_results)
max_f1 = threshold_results["f1"].max()

selected_threshold = threshold_results.loc[
    threshold_results["f1"] == max_f1,
    "threshold"
].min()

print("\nSelected threshold:", selected_threshold)
print("Best validation F1:", max_f1)
threshold_results.to_csv(
    REPORTS / "threshold_results.csv",
    index=False
)

print("Saved threshold results to:",
      REPORTS / "threshold_results.csv")
X_test_aug = np.column_stack([
    np.ones(len(X_test_scaled)),
    X_test_scaled
])

numpy_test_prob = sigmoid(X_test_aug @ beta)

sklearn_test_prob = sk_model.predict_proba(X_test_scaled)[:, 1]

baseline_test_prob = baseline.predict_proba(X_test_scaled)[:, 1]

print("\nsklearn classes:", sk_model.classes_)
print("Baseline classes:", baseline.classes_)
assignment_rows = []

def add_result(split_name, model_name, policy_name, threshold, y_true, probabilities):
    results = evaluate_model(y_true, probabilities, threshold)

    assignment_rows.append({
        "split": split_name,
        "model": model_name,
        "policy": policy_name,
        "threshold": threshold,
        "tn": results["tn"],
        "fp": results["fp"],
        "fn": results["fn"],
        "tp": results["tp"],
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "log_loss": results["log_loss"]
    })
    # Validation
add_result("validation", "baseline", "default", 0.50,
           y_val, baseline_val_prob)

add_result("validation", "numpy", "default", 0.50,
           y_val, numpy_val_prob)

add_result("validation", "numpy", "selected", selected_threshold,
           y_val, numpy_val_prob)

add_result("validation", "sklearn", "default", 0.50,
           y_val, sklearn_val_prob)

add_result("validation", "sklearn", "selected", selected_threshold,
           y_val, sklearn_val_prob)


# Test
add_result("test", "baseline", "default", 0.50,
           y_test, baseline_test_prob)

add_result("test", "numpy", "default", 0.50,
           y_test, numpy_test_prob)

add_result("test", "numpy", "selected", selected_threshold,
           y_test, numpy_test_prob)

add_result("test", "sklearn", "default", 0.50,
           y_test, sklearn_test_prob)

add_result("test", "sklearn", "selected", selected_threshold,
           y_test, sklearn_test_prob)
assignment_results = pd.DataFrame(assignment_rows)

print("\nAssignment results:")
print(assignment_results)

assignment_results.to_csv(
    REPORTS / "assignment_results.csv",
    index=False,
    float_format="%.6f"
)

print("Saved assignment results to:",
      REPORTS / "assignment_results.csv")

