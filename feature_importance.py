import pandas as pd
import joblib
import matplotlib.pyplot as plt

# File paths
MODEL_PATH = "src/data/models/risk_model.pkl"
SCALER_PATH = "src/data/models/scaler.pkl"
FEATURES_PATH = "src/data/models/features.pkl"

# Load model, scaler and feature names
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

# Get model coefficients
coefficients = model.coef_

# For multiclass Logistic Regression, calculate the
# average absolute coefficient across all classes
importance = abs(coefficients).mean(axis=0)

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Workplace Factor")
plt.title("Feature Importance - Occupational Safety Risk Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()