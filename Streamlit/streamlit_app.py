import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load trained Random Forest model
model = joblib.load("random_forest_concrete.pkl")  # replace with your .pkl file

# Streamlit app title
st.title("Concrete Compressive Strength Prediction")

st.markdown("""
This app predicts the **compressive strength of concrete (MPa)** based on mix components and curing age.
""")

# --- Define input sliders for features ---
cement = st.slider("Cement (kg/m³)", min_value=100, max_value=600, value=300)
slag = st.slider("Blast Furnace Slag (kg/m³)", min_value=0, max_value=400, value=50)
fly_ash = st.slider("Fly Ash (kg/m³)", min_value=0, max_value=200, value=20)
water = st.slider("Water (kg/m³)", min_value=100, max_value=250, value=180)
superplasticizer = st.slider("Superplasticizer (kg/m³)", min_value=0, max_value=35, value=5)
coarse_agg = st.slider("Coarse Aggregate (kg/m³)", min_value=600, max_value=1200, value=1000)
fine_agg = st.slider("Fine Aggregate (kg/m³)", min_value=500, max_value=1000, value=750)
age = st.slider("Age (days)", min_value=1, max_value=365, value=28)

# --- Predict button ---
if st.button("Predict Concrete Strength"):

    # Create a DataFrame with one row
    df_input = pd.DataFrame([[
        cement, slag, fly_ash, water, superplasticizer,
        coarse_agg, fine_agg, age
    ]], columns=model.feature_names_in_)

    # Predict
    y_pred = model.predict(df_input)[0]

    st.success(f"Predicted Concrete Compressive Strength: {y_pred:.2f} MPa")

# --- Page design ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://images.unsplash.com/photo-1581091870622-3b4216be55d2?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)
