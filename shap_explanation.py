import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

DATA_PATH = "src/data/construction_safety_data.csv"
MODEL_PATH = "src/data/models/risk_model.pkl"
SCALER_PATH = "src/data/models/scaler.pkl"
FEATURES_PATH = "src/data/models/features.pkl"

# Load dataset
data = pd.read_csv(DATA_PATH)

# Separate features and target
X = data.drop("risk_level", axis=1)

# Load trained model, scaler and feature names
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

# Scale the data
X_scaled = scaler.transform(X)

# Convert back to DataFrame with feature names
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

print("\n==============================")
print("SHAP EXPLAINABLE AI ANALYSIS")
print("==============================")

# Create SHAP explainer
explainer = shap.Explainer(model, X_scaled_df)

# Calculate SHAP values
shap_values = explainer(X_scaled_df)

print("SHAP analysis completed successfully!")

# SHAP summary plot
shap.summary_plot(
    shap_values,
    X_scaled_df,
    show=False
)

plt.title("SHAP Feature Importance - Occupational Safety Risk")

plt.tight_layout()
plt.show()