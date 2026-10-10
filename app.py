import streamlit as st
import pandas as pd
import pickle

# Configure the web application
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)

st.title("Heart Disease Prediction System")

st.write(
    "Enter the required patient information to obtain "
    "a prediction from the trained machine learning model."
)

# Load the trained SVC model
@st.cache_resource
def load_model():
    with open("svc_trained_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Take input from the user
st.subheader("Enter Patient Information")

cp_input = st.selectbox(
    "Chest Pain Type (cp)",
    options=[0, 1, 2, 3]
)

ca_input = st.selectbox(
    "Number of Major Vessels (ca)",
    options=[0, 1, 2, 3, 4]
)

oldpeak_input = st.number_input(
    "Oldpeak Value",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)

sex_input = st.selectbox(
    "Sex (sex)",
    options=[0, 1],
    format_func=lambda x: (
        "Female (0)" if x == 0 else "Male (1)"
    )
)

# Predict when the button is clicked
if st.button("Predict"):
    user_input = pd.DataFrame(
        [[cp_input, ca_input, oldpeak_input, sex_input]],
        columns=["cp", "ca", "oldpeak", "sex"]
    )

    prediction = model.predict(user_input)[0]

    if prediction == 1:
        st.warning(
            "Model Prediction: Heart disease present."
        )
    else:
        st.success(
            "Model Prediction: Heart disease not present."
        )

    st.caption(
        "This result is for educational purposes only "
        "and is not a medical diagnosis."
    )
