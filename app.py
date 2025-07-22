import streamlit as st
import pandas as pd
import pickle

# Load model and feature columns
with open('satisfaction_model.pkl', 'rb') as file:
    model, model_columns = pickle.load(file)

st.set_page_config(page_title="✈️ Flight Passenger Satisfaction Predictor")
st.title("✈️ Flight Passenger Satisfaction Predictor")
st.write("Fill the details below to predict whether the passenger is satisfied or not.")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
age = st.slider("Age", 7, 85, 30)
travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
class_type = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
flight_distance = st.number_input("Flight Distance", min_value=30, max_value=5000, value=500)

# Services (ratings from 0 to 5)
services = {}
service_features = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"
]

for feature in service_features:
    services[feature] = st.slider(feature, 0, 5, 3)

# Build input DataFrame
input_data = {
    "Gender": gender,
    "Customer Type": customer_type,
    "Age": age,
    "Type of Travel": travel_type,
    "Class": class_type,
    "Flight Distance": flight_distance,
    **services
}

input_df = pd.DataFrame([input_data])

# One-hot encode and reindex to match training features
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][prediction]

    label = "Satisfied 😊" if prediction == 1 else "Not Satisfied 😕"
    st.subheader(f"Prediction: {label}")
    st.write(f"Model Confidence: **{probability * 100:.2f}%**")
