import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ACCESSIBLE UI STYLE
# ============================================================
st.markdown("""
<style>
    :root {
        --navy: #17324D;
        --blue: #2563EB;
        --teal: #0F766E;
        --green: #15803D;
        --red: #B91C1C;
        --amber: #B45309;
        --bg: #F8FAFC;
        --text: #172033;
        --muted: #475569;
        --border: #CBD5E1;
        --white: #FFFFFF;
        --focus: #F59E0B;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    html, body, [class*="css"] {
        font-size: 17px;
    }

    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: -0.02em;
    }

    h1 { font-size: 2.35rem !important; }
    h2 { font-size: 1.65rem !important; }
    h3 { font-size: 1.25rem !important; }

    label {
        font-weight: 700 !important;
        color: var(--text) !important;
    }

    .hero {
        background: var(--navy);
        color: white;
        border-radius: 20px;
        padding: 1.7rem 1.8rem;
        margin-bottom: 1.25rem;
    }

    .hero h1, .hero p {
        color: white !important;
    }

    .hero p {
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(23, 50, 77, 0.06);
    }

    .result {
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1rem 0;
        border: 2px solid;
    }

    .result-positive {
        background: #FEF2F2;
        border-color: #B91C1C;
        color: #7F1D1D;
    }

    .result-negative {
        background: #ECFDF5;
        border-color: #15803D;
        color: #14532D;
    }

    .notice {
        background: #FFFBEB;
        border-left: 6px solid #B45309;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        color: #78350F;
    }

    .info-box {
        background: #EFF6FF;
        border-left: 6px solid #2563EB;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        color: #1E3A8A;
    }

    .stButton > button, .stDownloadButton > button {
        min-height: 50px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 16px;
    }

    *:focus-visible {
        outline: 4px solid var(--focus) !important;
        outline-offset: 3px !important;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border);
        padding: 1rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "model" not in st.session_state:
    st.session_state.model = None
if "model_info" not in st.session_state:
    st.session_state.model_info = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Risk Screening</h1>
    <p>
        Enter a patient's health information and the machine-learning model
        will estimate whether the patient's profile is predicted as
        <strong>Diabetes</strong> or <strong>No Diabetes</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
<strong>⚠️ Important:</strong> This application is a machine-learning
screening tool based on the model from the supplied notebook. It is
<strong>not a medical diagnosis</strong> and must not replace a doctor's
evaluation, laboratory testing, or professional medical advice.
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR: MODEL SETUP
# ============================================================
with st.sidebar:
    st.header("⚙️ Model Setup")

    st.write(
        "The supplied notebook trains a Logistic Regression pipeline using "
        "the diabetes prediction dataset."
    )

    training_file = st.file_uploader(
        "Upload training dataset",
        type=["csv"],
        help=(
            "Upload the same diabetes_prediction_dataset.csv used by the "
            "notebook. The CSV must contain the notebook's required columns."
        ),
    )

    st.divider()

    st.subheader("♿ Accessibility")
    st.write(
        "Large controls, readable contrast, keyboard focus indicators, "
        "clear labels, and text-based results are used so the interface "
        "does not depend on color alone."
    )

# ============================================================
# MODEL TRAINING
# ============================================================
required_columns = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
]

if training_file is not None:
    try:
        df = pd.read_csv(training_file)

        missing = [c for c in required_columns if c not in df.columns]

        if missing:
            st.error(
                "The uploaded dataset is missing these required columns: "
                + ", ".join(missing)
            )
            st.stop()

        # Same duplicate-removal step as the notebook.
        df = df.drop_duplicates()

        X = df.drop("diabetes", axis=1)
        y = df["diabetes"]

        categorical_features = ["gender", "smoking_history"]
        numerical_features = [
            "age",
            "hypertension",
            "heart_disease",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    numerical_features,
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        y_test_probability = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_probability)

        st.session_state.model = model
        st.session_state.model_info = {
            "rows": len(df),
            "accuracy": accuracy,
            "auc": auc,
        }

    except Exception as e:
        st.error(f"Could not train the model from this CSV: {e}")
        st.stop()

# ============================================================
# MODEL STATUS
# ============================================================
if st.session_state.model is None:
    st.markdown("""
    <div class="info-box">
    <strong>Step 1 — Prepare the model:</strong><br>
    Upload <strong>diabetes_prediction_dataset.csv</strong> in the
    left sidebar. The app will automatically train the same Logistic
    Regression pipeline used in your notebook.
    </div>
    """, unsafe_allow_html=True)

    st.header("What information will be entered?")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card">
        <strong>Basic information</strong>
        <ul>
            <li>Gender</li>
            <li>Age</li>
            <li>Smoking history</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <strong>Health information</strong>
        <ul>
            <li>Hypertension</li>
            <li>Heart disease</li>
            <li>BMI</li>
            <li>HbA1c level</li>
            <li>Blood glucose level</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# MODEL READY
# ============================================================
info = st.session_state.model_info

st.success("✓ Model is ready. You can now enter patient information.")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Training records", f"{info['rows']:,}")
with m2:
    st.metric("Test accuracy", f"{info['accuracy'] * 100:.1f}%")
with m3:
    st.metric("ROC-AUC", f"{info['auc']:.3f}")

# ============================================================
# PATIENT FORM
# ============================================================
st.header("1️⃣ Enter patient information")

st.markdown("""
<div class="info-box">
Please enter the patient's information carefully. The prediction is based
on the values entered here and the model trained from the supplied dataset.
</div>
""", unsafe_allow_html=True)

with st.form("patient_form"):
    st.subheader("👤 Basic information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
            help="Select the patient's recorded gender category.",
        )

    with col2:
        age = st.number_input(
            "Age (years)",
            min_value=0.0,
            max_value=120.0,
            value=45.0,
            step=1.0,
        )

    with col3:
        smoking_history = st.selectbox(
            "Smoking history",
            ["never", "former", "current", "No Info", "ever", "not current"],
            help="Choose the patient's smoking-history category.",
        )

    st.subheader("🫀 Health measurements")

    col1, col2 = st.columns(2)

    with col1:
        hypertension_label = st.radio(
            "Hypertension",
            ["No", "Yes"],
            horizontal=True,
            help="Select Yes if the patient has hypertension.",
        )

        heart_disease_label = st.radio(
            "Heart disease",
            ["No", "Yes"],
            horizontal=True,
            help="Select Yes if the patient has heart disease.",
        )

        bmi = st.number_input(
            "BMI",
            min_value=5.0,
            max_value=100.0,
            value=27.5,
            step=0.1,
            help="Enter the patient's body mass index.",
        )

    with col2:
        hba1c = st.number_input(
            "HbA1c level (%)",
            min_value=2.0,
            max_value=20.0,
            value=5.8,
            step=0.1,
            help="Enter the HbA1c value recorded for the patient.",
        )

        glucose = st.number_input(
            "Blood glucose level",
            min_value=20.0,
            max_value=1000.0,
            value=120.0,
            step=1.0,
            help="Enter the blood glucose measurement using the same unit as the training dataset.",
        )

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes Status",
        type="primary",
        use_container_width=True,
    )

# ============================================================
# PREDICTION
# ============================================================
if submitted:
    patient = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [1 if hypertension_label == "Yes" else 0],
        "heart_disease": [1 if heart_disease_label == "Yes" else 0],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose],
    })

    model = st.session_state.model

    try:
        prediction = int(model.predict(patient)[0])
        probability = float(model.predict_proba(patient)[0][1])

        st.session_state.prediction = {
            "prediction": prediction,
            "probability": probability,
            "patient": patient,
        }
    except Exception as e:
        st.error(f"Prediction could not be completed: {e}")
        st.stop()

# ============================================================
# RESULT
# ============================================================
if st.session_state.prediction is not None:
    result = st.session_state.prediction
    prediction = result["prediction"]
    probability = result["probability"]

    st.header("2️⃣ Prediction result")

    if prediction == 1:
        st.markdown(f"""
        <div class="result result-positive">
            <h2>⚠️ Model prediction: Diabetes</h2>
            <p>
                The model classified this patient profile as
                <strong>Diabetes</strong>.
            </p>
            <h3>Estimated model probability: {probability * 100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result result-negative">
            <h2>✅ Model prediction: No Diabetes</h2>
            <p>
                The model classified this patient profile as
                <strong>No Diabetes</strong>.
            </p>
            <h3>Estimated model probability: {probability * 100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="notice">
    <strong>Clinical caution:</strong> The probability shown above is the
    model's estimated probability for this dataset and model. It is not
    the patient's clinical probability of having diabetes. A prediction of
    either class should be interpreted by a qualified healthcare professional
    using appropriate clinical assessment and diagnostic testing.
    </div>
    """, unsafe_allow_html=True)

    # Patient values
    with st.expander("View entered patient information"):
        display_patient = result["patient"].T
        display_patient.columns = ["Entered value"]
        st.dataframe(display_patient, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "Diabetes Risk Screening • Logistic Regression • "
    "Based on the supplied diabetes prediction notebook • "
    "For educational/research screening use only."
)
