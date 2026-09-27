import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="OHS AI Risk Assessment",
    page_icon="🦺",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = "construction_safety_data.csv"
MODEL_PATH = "risk_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)


# =========================================================
# LOAD DATA FOR SHAP
# =========================================================

data = pd.read_csv(DATA_PATH)

X_background = data.drop(
    "risk_level",
    axis=1
)

X_background = X_background[features]

X_background_scaled = scaler.transform(
    X_background
)

X_background_scaled = pd.DataFrame(
    X_background_scaled,
    columns=features
)

# Use 100 observations as SHAP background
shap_background = shap.sample(
    X_background_scaled,
    100,
    random_state=42
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "🦺 AI-Based Occupational Safety Risk Assessment"
)

st.write(
    "This system uses Machine Learning and Explainable AI "
    "to predict occupational safety risks in construction "
    "workplaces."
)

st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.header("🏗️ Workplace Risk Assessment")

st.write(
    "Enter the workplace conditions below and click "
    "**Predict Safety Risk**."
)


# =========================================================
# THREE COLUMNS
# =========================================================

col1, col2, col3 = st.columns(3)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=15.0,
        max_value=45.0,
        value=30.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=20.0,
        max_value=90.0,
        value=60.0
    )

    noise_db = st.number_input(
        "Noise Level (dB)",
        min_value=60.0,
        max_value=110.0,
        value=80.0
    )

    shift_hours = st.number_input(
        "Shift Hours",
        min_value=6.0,
        max_value=12.0,
        value=8.0
    )

    break_hours = st.number_input(
        "Break Hours",
        min_value=0.25,
        max_value=2.0,
        value=1.0
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    ppe_compliance = st.slider(
        "PPE Compliance (%)",
        min_value=50,
        max_value=100,
        value=85
    )

    experience_years = st.number_input(
        "Experience (Years)",
        min_value=0.0,
        max_value=20.0,
        value=5.0
    )

    fatigue_level = st.slider(
        "Fatigue Level (1-10)",
        min_value=1,
        max_value=10,
        value=5
    )

    previous_incidents = st.number_input(
        "Previous Incidents",
        min_value=0,
        max_value=10,
        value=1
    )

    near_misses = st.number_input(
        "Near Misses",
        min_value=0,
        max_value=15,
        value=2
    )


# =========================================================
# COLUMN 3
# =========================================================

with col3:

    training_hours = st.number_input(
        "Training Hours",
        min_value=0.0,
        max_value=40.0,
        value=10.0
    )

    equipment_age = st.number_input(
        "Equipment Age (Years)",
        min_value=0.0,
        max_value=20.0,
        value=5.0
    )

    maintenance_score = st.slider(
        "Maintenance Score",
        min_value=40,
        max_value=100,
        value=80
    )

    workload = st.slider(
        "Workload (%)",
        min_value=30,
        max_value=100,
        value=60
    )

    work_at_height = st.selectbox(
        "Work at Height",
        ["No", "Yes"]
    )

    heavy_machinery = st.selectbox(
        "Heavy Machinery",
        ["No", "Yes"]
    )


# =========================================================
# CONVERT YES / NO TO 0 / 1
# =========================================================

work_at_height_value = (
    1 if work_at_height == "Yes" else 0
)

heavy_machinery_value = (
    1 if heavy_machinery == "Yes" else 0
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Safety Risk",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

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

        "work_at_height": work_at_height_value,

        "heavy_machinery": heavy_machinery_value

    }])


    # -----------------------------------------------------
    # ENSURE CORRECT FEATURE ORDER
    # -----------------------------------------------------

    input_data = input_data[features]


    # -----------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )

    input_scaled_df = pd.DataFrame(
        input_scaled,
        columns=features
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        input_scaled_df
    )[0]


    # -----------------------------------------------------
    # PREDICTION PROBABILITIES
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        input_scaled_df
    )[0]


    # -----------------------------------------------------
    # RISK NAMES
    # -----------------------------------------------------

    risk_names = {
        0: "LOW RISK",
        1: "MEDIUM RISK",
        2: "HIGH RISK"
    }

    predicted_risk = risk_names[prediction]


    # =====================================================
    # SHAP EXPLANATION
    # =====================================================

    explainer = shap.Explainer(
        model,
        shap_background
    )

    shap_values = explainer(
        input_scaled_df
    )

    values = shap_values.values


    # -----------------------------------------------------
    # GET CONTRIBUTION FOR PREDICTED CLASS
    # -----------------------------------------------------

    if values.ndim == 3:

        contribution = values[
            0,
            :,
            prediction
        ]

    else:

        contribution = values[0]


    # -----------------------------------------------------
    # CREATE SHAP TABLE
    # -----------------------------------------------------

    explanation_df = pd.DataFrame({

        "Feature": features,

        "Contribution": contribution

    })


    explanation_df[
        "Absolute_Contribution"
    ] = explanation_df[
        "Contribution"
    ].abs()


    explanation_df = explanation_df.sort_values(
        by="Absolute_Contribution",
        ascending=False
    )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.divider()

    st.header(
        "📊 Risk Assessment Result"
    )


    # -----------------------------------------------------
    # RISK MESSAGE
    # -----------------------------------------------------

    if prediction == 0:

        st.success(
            "🟢 LOW RISK\n\n"
            "The workplace conditions indicate a "
            "relatively low predicted safety risk."
        )

    elif prediction == 1:

        st.warning(
            "🟡 MEDIUM RISK\n\n"
            "The workplace conditions indicate a "
            "moderate predicted safety risk."
        )

    else:

        st.error(
            "🔴 HIGH RISK\n\n"
            "The workplace conditions indicate a "
            "high predicted safety risk."
        )


    # =====================================================
    # PROBABILITIES
    # =====================================================

    st.subheader(
        "📈 Prediction Probabilities"
    )

    probability_data = pd.DataFrame({

        "Risk Level": [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],

        "Probability": probabilities * 100

    })


    st.bar_chart(
        probability_data.set_index(
            "Risk Level"
        )
    )


    st.write(
        "The probabilities represent the model's "
        "estimated probability for each risk category."
    )


    # =====================================================
    # SHAP EXPLANATION
    # =====================================================

    st.divider()

    st.subheader(
        "🔎 Why did the model make this prediction?"
    )

    st.write(
        "SHAP identifies the workplace factors that "
        "contributed most strongly to this individual "
        "risk prediction."
    )


    # -----------------------------------------------------
    # TOP 7 FACTORS
    # -----------------------------------------------------

    top_factors = explanation_df.head(7).copy()


    # Sort for chart
    chart_data = top_factors.sort_values(
        by="Contribution"
    )


    # -----------------------------------------------------
    # SHAP BAR CHART
    # -----------------------------------------------------

    st.subheader(
        "Top Contributing Factors"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        chart_data["Feature"],
        chart_data["Contribution"]
    )

    ax.axvline(
        x=0,
        linestyle="--"
    )

    ax.set_xlabel(
        "SHAP Contribution"
    )

    ax.set_ylabel(
        "Workplace Factor"
    )

    ax.set_title(
        "Individual Risk Explanation - "
        + predicted_risk
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # =====================================================
    # SHAP TABLE
    # =====================================================

    st.subheader(
        "📋 Main Contributing Factors"
    )

    display_table = explanation_df.head(7)[
        ["Feature", "Contribution"]
    ].copy()


    display_table[
        "Contribution"
    ] = display_table[
        "Contribution"
    ].round(3)


    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # SHAP INTERPRETATION
    # =====================================================

    st.info(
        "Positive SHAP values indicate that a factor "
        "contributed toward the predicted risk class, "
        "while negative values indicate contribution "
        "away from the predicted risk class."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Predictive Risk Assessment Framework "
    "for Occupational Safety in Construction Workplaces"
)
# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.header("📌 About the AI Risk Assessment System")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Dataset Size",
        "3,000"
    )

with col2:
    st.metric(
        "Input Features",
        "16"
    )

with col3:
    st.metric(
        "Model Accuracy",
        "90.17%"
    )

with col4:
    st.metric(
        "Best Model",
        "Logistic Regression"
    )

st.write(
    """
    ### System Overview

    This application uses a machine-learning model to
    estimate occupational safety risk based on workplace
    and worker-related factors.

    The system classifies observations into three risk
    categories:

    - 🟢 Low Risk
    - 🟡 Medium Risk
    - 🔴 High Risk

    Explainable AI using SHAP is integrated to identify
    the factors that contribute most strongly to each
    individual prediction.
    """
)

st.warning(
    "Research note: The current dataset is synthetically "
    "generated for research and demonstration purposes. "
    "The model should therefore not be interpreted as a "
    "validated real-world accident prediction system."
)
