import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Carbon Emission Calculator", layout="centered")

# Load model and metadata
@st.cache_resource
def load_model():
    return joblib.load("carbon_model.pkl")

@st.cache_resource
def load_metadata():
    with open("feature_metadata.json", "r") as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

st.title("🌍 Carbon Emission Calculator")
st.write("Enter your lifestyle details to estimate your carbon footprint.")

# Build input form dynamically
user_input = {}

with st.form("carbon_form"):
    st.subheader("📝 Your Information")

    # Categorical Features (Dropdowns)
    for col, options in metadata["categorical"].items():
        user_input[col] = st.selectbox(col, options)

    # Numeric Features (Sliders)
    for col, rng in metadata["numeric"].items():
        user_input[col] = st.number_input(
            col,
            min_value=rng["min"],
            max_value=rng["max"],
            value=rng["min"]
        )

    submit = st.form_submit_button("Calculate Emission")

# Prediction
if submit:
    try:
        input_df = pd.DataFrame([user_input])
        carbon_pred = model.predict(input_df)[0]

        st.success(f"✅ Estimated Carbon Emission: **{carbon_pred:.2f}**")

        # Basic interpretation (temporary)
        if carbon_pred < 200:
            st.info("🌱 Impact Level: **Low**")
        elif carbon_pred < 300:
            st.warning("⚠️ Impact Level: **Medium**")
        else:
            st.error("🔥 Impact Level: **High**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
