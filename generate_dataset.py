import pandas as pd
import numpy as np

np.random.seed(42)

# Number of observations
n = 3000

# -----------------------------
# Generate workplace data
# -----------------------------

temperature = np.random.uniform(15, 45, n)
humidity = np.random.uniform(20, 90, n)
noise_db = np.random.uniform(60, 110, n)

shift_hours = np.random.uniform(6, 12, n)
break_hours = np.random.uniform(0.25, 2, n)

ppe_compliance = np.random.uniform(50, 100, n)
experience_years = np.random.uniform(0, 20, n)

previous_incidents = np.random.poisson(1, n)
near_misses = np.random.poisson(2, n)

training_hours = np.random.uniform(0, 40, n)
equipment_age = np.random.uniform(0, 20, n)
maintenance_score = np.random.uniform(40, 100, n)

workload = np.random.uniform(30, 100, n)

work_at_height = np.random.randint(0, 2, n)
heavy_machinery = np.random.randint(0, 2, n)

fatigue_level = np.random.uniform(1, 10, n)

# -----------------------------
# Calculate safety risk score
# -----------------------------

risk_score = (
    temperature * 0.05
    + humidity * 0.02
    + noise_db * 0.08
    + shift_hours * 1.5
    - break_hours * 2
    - ppe_compliance * 0.08
    - experience_years * 0.5
    + previous_incidents * 3
    + near_misses * 2
    - training_hours * 0.05
    + equipment_age * 0.8
    - maintenance_score * 0.04
    + workload * 0.08
    + work_at_height * 5
    + heavy_machinery * 4
    + fatigue_level * 2
)

# Add random variation
risk_score += np.random.normal(0, 2, n)

# -----------------------------
# Create balanced risk classes
# -----------------------------

risk_level = pd.qcut(
    risk_score,
    q=3,
    labels=[1, 2, 3]
)

# -----------------------------
# Create DataFrame
# -----------------------------

data = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "noise_db": noise_db,
    "shift_hours": shift_hours,
    "break_hours": break_hours,
    "ppe_compliance": ppe_compliance,
    "experience_years": experience_years,
    "fatigue_level": fatigue_level,
    "previous_incidents": previous_incidents,
    "near_misses": near_misses,
    "training_hours": training_hours,
    "equipment_age": equipment_age,
    "maintenance_score": maintenance_score,
    "workload": workload,
    "work_at_height": work_at_height,
    "heavy_machinery": heavy_machinery,
    "risk_level": risk_level.astype(int)
})

# -----------------------------
# Save dataset
# -----------------------------

data.to_csv(
    "src/data/construction_safety_data.csv",
    index=False
)

print("==============================")
print("DATASET GENERATED SUCCESSFULLY")
print("==============================")

print("Dataset shape:", data.shape)

print("\nRisk Level Distribution:")
print(data["risk_level"].value_counts().sort_index())

print("\nDataset saved to:")
print("src/data/construction_safety_data.csv")