import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==============================
# FILE PATHS
# ==============================

DATA_PATH = "src/data/construction_safety_data.csv"
MODEL_PATH = "src/data/models/risk_model.pkl"
SCALER_PATH = "src/data/models/scaler.pkl"
FEATURES_PATH = "src/data/models/features.pkl"

# ==============================
# LOAD DATA AND MODEL
# ==============================

data = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

# Workplace features
X = data.drop("risk_level", axis=1)

# Make sure feature order is correct
X = X[features]

# Scale complete dataset
X_scaled = scaler.transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

# ==============================
# SAMPLE WORKPLACE OBSERVATION
# ==============================

sample = pd.DataFrame([{
    "temperature": 35,
    "humidity": 70,
    "noise_db": 95,
    "shift_hours": 10,
    "break_hours": 0.5,
    "ppe_compliance": 70,
    "experience_years": 3,
    "fatigue_level": 8,
    "previous_incidents": 3,
    "near_misses": 4,
    "training_hours": 10,
    "equipment_age": 15,
    "maintenance_score": 60,
    "workload": 85,
    "work_at_height": 1,
    "heavy_machinery": 1
}])

sample = sample[features]

# Scale individual sample
sample_scaled = scaler.transform(sample)

sample_scaled_df = pd.DataFrame(
    sample_scaled,
    columns=features
)

# ==============================
# PREDICT RISK
# ==============================

prediction = model.predict(sample_scaled_df)[0]

risk_names = {
    0: "LOW RISK",
    1: "MEDIUM RISK",
    2: "HIGH RISK"
}

predicted_risk = risk_names[prediction]

print("\n==============================")
print("INDIVIDUAL RISK PREDICTION")
print("==============================")

print("Predicted Risk:", predicted_risk)

# ==============================
# SHAP BACKGROUND DATA
# ==============================

# Use 100 normal workplace observations
# as the SHAP background/reference dataset.

background = shap.sample(
    X_scaled_df,
    100,
    random_state=42
)

# ==============================
# CREATE SHAP EXPLAINER
# ==============================

explainer = shap.Explainer(
    model,
    background
)

# Explain the individual workplace sample
shap_values = explainer(sample_scaled_df)

# ==============================
# GET SHAP VALUES
# ==============================

values = shap_values.values

print("\nSHAP value shape:", values.shape)

# Multiclass Logistic Regression
if values.ndim == 3:

    # Select the predicted risk class
    contribution = values[0, :, prediction]

else:

    contribution = values[0]

# ==============================
# CREATE EXPLANATION TABLE
# ==============================

explanation_df = pd.DataFrame({
    "Feature": features,
    "Contribution": contribution
})

explanation_df["Absolute_Contribution"] = (
    explanation_df["Contribution"].abs()
)

explanation_df = explanation_df.sort_values(
    by="Absolute_Contribution",
    ascending=False
)

print("\n==============================")
print("TOP FACTORS AFFECTING PREDICTION")
print("==============================")

print(
    explanation_df[
        ["Feature", "Contribution"]
    ].head(10).to_string(index=False)
)

# ==============================
# PLOT
# ==============================

top_features = explanation_df.head(10)

top_features = top_features.sort_values(
    by="Contribution"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Contribution"]
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.xlabel("SHAP Contribution")
plt.ylabel("Workplace Factor")

plt.title(
    "Individual Risk Explanation - " + predicted_risk
)

plt.tight_layout()

plt.show()