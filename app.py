from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "water_potability_pipeline.joblib"
DATA_PATH = ROOT / "data" / "water_potability.csv"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)

st.title("💧 Water Potability Prediction System")
st.write(
    "Enter the measured water-quality parameters below. "
    "The trained machine-learning model will estimate whether the sample is potable."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python src/train_model.py` first.")
    st.stop()

model = load_model()
reference = load_reference_data()

with st.sidebar:
    st.header("About")
    st.write(
        "This application uses a neural-network regression model and a 0.5 "
        "decision threshold to classify water samples as potable or not potable."
    )

st.subheader("Water quality parameters")
st.caption("Default values are dataset medians and can be edited.")

values = {}
cols = st.columns(3)

for i, feature in enumerate(FEATURES):
    series = reference[feature].dropna()
    default = float(series.median())
    min_v = float(series.min())
    max_v = float(series.max())

    with cols[i % 3]:
        values[feature] = st.number_input(
            feature.replace("_", " "),
            min_value=min_v,
            max_value=max_v,
            value=default,
            format="%.4f",
            help=f"Observed dataset range: {min_v:.3f} – {max_v:.3f}"
        )

st.divider()

if st.button("Analyze Water Quality", type="primary", use_container_width=True):
    sample = pd.DataFrame([values], columns=FEATURES)
    raw_score = float(model.predict(sample)[0])
    display_score = max(0.0, min(1.0, raw_score))
    prediction = int(raw_score >= 0.5)

    left, right = st.columns([2, 1])
    with left:
        if prediction == 1:
            st.success("### 💧 Predicted: POTABLE")
            st.write("The model classifies this water sample as potable.")
        else:
            st.error("### ⚠️ Predicted: NOT POTABLE")
            st.write("The model classifies this water sample as not potable.")

    with right:
        st.metric("Model score", f"{display_score:.3f}")
        st.caption("Decision threshold: 0.500")

    with st.expander("View entered parameters"):
        st.dataframe(sample, use_container_width=True)

st.divider()
st.caption(
    "Educational machine-learning project. A prediction from this model should "
    "not be used to determine whether water is medically or legally safe to drink."
)
