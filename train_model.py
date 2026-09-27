import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# ==============================
# 1. File paths
# ==============================

DATA_PATH = "src/data/construction_safety_data.csv"
MODEL_DIR = "src/data/models"

# Create models folder if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================
# 2. Load dataset
# ==============================

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==============================
# 3. Separate features and target
# ==============================

X = data.drop("risk_level", axis=1)
y = data["risk_level"] - 1


# ==============================
# 4. Split dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 5. Scale the data
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================
# 6. Create models
# ==============================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )
}


# ==============================
# 7. Train and compare models
# ==============================

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# ==============================
# 8. Display results
# ==============================

results_df = pd.DataFrame(results)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df)


# ==============================
# 9. Select best model
# ==============================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

print("\nBest Model:", best_model_name)


# ==============================
# 10. Train final XGBoost model
# ==============================

final_model = models[best_model_name]

print("\nTraining final XGBoost model...")

final_model.fit(X_train_scaled, y_train)


# ==============================
# 11. Save model
# ==============================

joblib.dump(
    final_model,
    os.path.join(MODEL_DIR, "risk_model.pkl")
)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "scaler.pkl")
)

joblib.dump(
    list(X.columns),
    os.path.join(MODEL_DIR, "features.pkl")
)


# ==============================
# 12. Finish
# ==============================

print("\n==============================")
print("TRAINING COMPLETED!")
print("==============================")

print("Model saved as:")
print("src/data/models/risk_model.pkl")

print("Scaler saved as:")
print("src/data/models/scaler.pkl")

print("Features saved as:")
print("src/data/models/features.pkl")