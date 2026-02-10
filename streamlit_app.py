import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load trained Random Forest model
model_file = "random_forest_concrete.pkl"
if not os.path.exists(model_file):
    st.error(f"Model file '{model_file}' not found.")
    st.stop()

model = joblib.load(model_file)  # load .pkl file

# Streamlit app title
st.title("Concrete Compressive Strength Prediction")

# --- About the model ---
st.subheader("About the Model")
st.markdown("""
This app uses a **Random Forest Regressor** trained on concrete mix data to predict **compressive strength (in MPa)**. 
The model considers **8 key features** of the concrete mixture:

- **Cement, Blast Furnace Slag, Fly Ash**: Main binder materials.
- **Water**: Affects hydration and strength.
- **Superplasticizer**: Improves workability without adding more water.
- **Coarse and Fine Aggregate**: Provide volume and affect strength.
- **Age**: Number of days the concrete has cured.

By entering the quantities of each component and the curing age, the model estimates the concrete strength. 
This helps engineers **save time during mix design and testing phases**, allowing quick evaluation of different mix scenarios without performing physical tests.
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

