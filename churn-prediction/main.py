"""
Main entry point: runs the full churn prediction pipeline end-to-end.

    python main.py

Steps: load & preprocess data -> train 3 models -> evaluate & compare ->
save plots -> save the best model to disk.
"""
import os
import pickle

from src.preprocessing import prepare_data
from src.train_models import train_all_models
from src.evaluate import evaluate_all_models, plot_confusion_matrices, plot_feature_importance

DATA_PATH = "data/customer_churn.csv"
MODEL_SAVE_PATH = "models/best_model.pkl"


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"No dataset found at {DATA_PATH}.")
        print("Run: python src/generate_data.py   (or drop the real Telco Churn CSV there)")
        return

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, encoders, scaler = prepare_data(DATA_PATH)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Churn rate in training set: {y_train.mean():.2%}")

    print("\nTraining models: Logistic Regression, Random Forest, XGBoost...")
    models = train_all_models(X_train, y_train)

    print("\nEvaluating models...")
    results_df, confusion_matrices = evaluate_all_models(models, X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    plot_confusion_matrices(confusion_matrices)

    for name, model in models.items():
        plot_feature_importance(model, X_train.columns, name)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    print(f"\nBest model by ROC-AUC: {best_model_name}")

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "encoders": encoders}, f)
    print(f"Saved best model to {MODEL_SAVE_PATH}")

    results_df.to_csv("outputs/model_comparison.csv", index=False)
    print("Saved comparison table to outputs/model_comparison.csv")


if __name__ == "__main__":
    main()
