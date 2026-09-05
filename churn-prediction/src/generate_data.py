"""
Generates a synthetic customer dataset with the same structure as the
well-known Telco Customer Churn dataset, so the project runs end-to-end
without needing to download anything externally.

If you have the real Kaggle "Telco Customer Churn" CSV, just drop it in
data/customer_churn.csv with the same column names and skip this script.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # number of synthetic customers


def generate_dataset(n=N) -> pd.DataFrame:
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    tenure = np.random.randint(0, 73, size=n)
    monthly_charges = np.round(np.random.uniform(18, 120, size=n), 2)
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.20]
    )
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
    )

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], size=n),
        "SeniorCitizen": np.random.choice([0, 1], size=n, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes", "No"], size=n),
        "Dependents": np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": np.random.choice(["Yes", "No"], size=n, p=[0.9, 0.1]),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], size=n),
        "InternetService": internet_service,
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "Contract": contract,
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n),
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
    })

    df["TotalCharges"] = np.round(df["tenure"] * df["MonthlyCharges"] * np.random.uniform(0.9, 1.0, size=n), 2)

    # --- Build churn probability so the data has realistic, learnable signal ---
    churn_prob = 0.10
    churn_prob = churn_prob + np.where(df["Contract"] == "Month-to-month", 0.35, 0)
    churn_prob = churn_prob + np.where(df["Contract"] == "One year", 0.05, 0)
    churn_prob = churn_prob + np.where(df["InternetService"] == "Fiber optic", 0.15, 0)
    churn_prob = churn_prob + np.where(df["tenure"] < 6, 0.20, 0)
    churn_prob = churn_prob - np.where(df["tenure"] > 48, 0.15, 0)
    churn_prob = churn_prob + np.where(df["PaymentMethod"] == "Electronic check", 0.10, 0)
    churn_prob = churn_prob + np.where(df["TechSupport"] == "No", 0.08, 0)
    churn_prob = np.clip(churn_prob, 0.02, 0.9)

    df["Churn"] = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

    # Introduce a few realistic missing values in TotalCharges (common in the real dataset)
    missing_idx = np.random.choice(df.index, size=int(0.01 * n), replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv("data/customer_churn.csv", index=False)
    print(f"Saved {len(dataset)} rows to data/customer_churn.csv")
    print(f"Churn rate: {(dataset['Churn'] == 'Yes').mean():.2%}")
