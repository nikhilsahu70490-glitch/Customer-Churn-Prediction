"""
Evaluates trained models on the test set using multiple metrics.

Accuracy alone is misleading on churn data because the classes are
usually imbalanced (far more "stayed" than "churned" customers) -
a model that just predicts "No churn" for everyone can still score
high accuracy while being useless. Precision/Recall/F1/ROC-AUC give
a fuller picture, and recall in particular matters most here since
missing an actual churner is usually costlier than a false alarm.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'='*50}\n{model_name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics, confusion_matrix(y_test, y_pred)


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    results = []
    confusion_matrices = {}

    for name, model in models.items():
        metrics, cm = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        confusion_matrices[name] = cm

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    return results_df, confusion_matrices


def plot_confusion_matrices(confusion_matrices: dict, save_path: str = "outputs/confusion_matrices.png"):
    n = len(confusion_matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"],
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved confusion matrices to {save_path}")


def plot_feature_importance(model, feature_names, model_name: str, save_path: str = None):
    """Works for tree-based models (Random Forest, XGBoost) that expose feature_importances_."""
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(10)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", color="steelblue")
    plt.title(f"Top 10 Feature Importances - {model_name}")
    plt.tight_layout()

    if save_path is None:
        save_path = f"outputs/feature_importance_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved feature importance plot to {save_path}")
