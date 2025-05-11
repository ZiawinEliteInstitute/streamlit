import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# App Title
st.markdown("<h1 style='text-align: center; color: white;'>🚢 Titanic Survival Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter passenger details below to predict survival chances</p>", unsafe_allow_html=True)

st.write("---")

# Create two columns for inputs
left_col, right_col = st.columns(2)

user_input = []

for i, col in enumerate(feature_columns):
    with (left_col if i % 2 == 0 else right_col):
        val = st.number_input(f"{col}", format="%.2f")
        user_input.append(val)

st.write("---")

# Predict Button
if st.button("Predict Survival"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    result_text = "🎉 Survived" if prediction[0] == 1 else "💀 Did Not Survive"

    st.markdown(
        f"<div style='text-align:center; padding:20px; border-radius:10px; background-color:#e6f2ff;'>"
        f"<h2 style='color:black;'>Prediction: {result_text}</h2></div>",
        unsafe_allow_html=True
    )


# Footer
st.write(" ")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
