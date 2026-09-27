import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# File paths
# -----------------------------
DATA_PATH = "src/data/construction_safety_data.csv"
MODEL_PATH = "src/data/models/risk_model.pkl"
SCALER_PATH = "src/data/models/scaler.pkl"

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv(DATA_PATH)

X = data.drop("risk_level", axis=1)
y = data["risk_level"] - 1

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Scale test data
# -----------------------------
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Make predictions
# -----------------------------
y_pred = model.predict(X_test_scaled)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("Accuracy:", round(accuracy, 4))

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Display Confusion Matrix
# -----------------------------
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Risk Level")
plt.ylabel("Actual Risk Level")
plt.title("Confusion Matrix - Construction Safety Risk Prediction")

plt.show()