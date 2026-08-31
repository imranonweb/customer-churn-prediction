import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src import config


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("MODEL OVERFITTING / UNDERFITTING CHECK")
print("=" * 70)


print("\nLoading datasets...")

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape : {test_df.shape}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load("artifacts/random_forest.pkl")

print("Model loaded successfully.")


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train_df.drop(columns=[config.TARGET])
y_train = train_df[config.TARGET]

X_test = test_df.drop(columns=[config.TARGET])
y_test = test_df[config.TARGET]


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_model(X, y, dataset_name):

    # Predictions
    y_pred = model.predict(X)

    # Probabilities
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_prob),
        "PR-AUC": average_precision_score(y, y_prob)
    }

    print("\n" + "-" * 70)
    print(f"{dataset_name} PERFORMANCE")
    print("-" * 70)

    for metric, value in metrics.items():
        print(f"{metric:<15}: {value:.4f}")

    return metrics


# ============================================================
# EVALUATE TRAINING DATA
# ============================================================

train_metrics = evaluate_model(
    X_train,
    y_train,
    "TRAINING DATA"
)


# ============================================================
# EVALUATE TEST DATA
# ============================================================

test_metrics = evaluate_model(
    X_test,
    y_test,
    "TEST DATA"
)


# ============================================================
# COMPARE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TRAIN VS TEST COMPARISON")
print("=" * 70)

print(
    f"\n{'Metric':<15}"
    f"{'Train':>12}"
    f"{'Test':>12}"
    f"{'Gap':>12}"
)

print("-" * 55)

for metric in train_metrics:

    train_value = train_metrics[metric]
    test_value = test_metrics[metric]

    gap = train_value - test_value

    print(
        f"{metric:<15}"
        f"{train_value:>12.4f}"
        f"{test_value:>12.4f}"
        f"{gap:>12.4f}"
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

roc_gap = train_metrics["ROC-AUC"] - test_metrics["ROC-AUC"]
pr_gap = train_metrics["PR-AUC"] - test_metrics["PR-AUC"]

print(f"\nROC-AUC Gap: {roc_gap:.4f}")
print(f"PR-AUC Gap : {pr_gap:.4f}")

print("\nHow to interpret:")

if roc_gap > 0.10:
    print(
        "\n⚠ Possible overfitting detected."
    )
    print(
        "Training performance is significantly higher than testing performance."
    )

elif train_metrics["ROC-AUC"] < 0.70 and test_metrics["ROC-AUC"] < 0.70:
    print(
        "\n⚠ Possible underfitting detected."
    )
    print(
        "Both training and testing performance are relatively low."
    )

else:
    print(
        "\n✓ No strong evidence of overfitting or underfitting."
    )
    print(
        "Training and testing performance are reasonably close."
    )
    print(
        "The model appears to generalize reasonably well."
    )


print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)