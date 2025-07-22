import streamlit as st
import pandas as pd
import pickle

# Load pipeline
with open("model_pipelines.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.title("✈️ Flight Passenger Satisfaction Predictor")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
age = st.slider("Age", 7, 100, 25)
type_of_travel = st.selectbox("Type of Travel", ["Personal Travel", "Business Travel"])
travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
flight_distance = st.slider("Flight Distance", 50, 5000, 500)

# Service Ratings
wifi = st.slider("Inflight wifi service", 0, 5)
time_convenient = st.slider("Departure/Arrival time convenient", 0, 5)
online_booking = st.slider("Ease of Online booking", 0, 5)
food = st.slider("Food and drink", 0, 5)
seat = st.slider("Seat comfort", 0, 5)
entertainment = st.slider("Inflight entertainment", 0, 5)
onboard_service = st.slider("On-board service", 0, 5)
legroom = st.slider("Leg room service", 0, 5)
baggage = st.slider("Baggage handling", 0, 5)
checkin = st.slider("Checkin service", 0, 5)
inflight_service = st.slider("Inflight service", 0, 5)
cleanliness = st.slider("Cleanliness", 0, 5)
online_boarding = st.slider("Online boarding", 0, 5)

# Delay Times
dep_delay = st.number_input("Departure Delay in Minutes", 0, 1000, 0)
arr_delay = st.number_input("Arrival Delay in Minutes", 0, 1000, 0)

# Prediction
if st.button("Predict Satisfaction"):
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": type_of_travel,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Inflight wifi service": wifi,
        "Departure/Arrival time convenient": time_convenient,
        "Ease of Online booking": online_booking,
        "Food and drink": food,
        "Seat comfort": seat,
        "Inflight entertainment": entertainment,
        "On-board service": onboard_service,
        "Leg room service": legroom,
        "Baggage handling": baggage,
        "Checkin service": checkin,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Online boarding": online_boarding,
        "Departure Delay in Minutes": dep_delay,
        "Arrival Delay in Minutes": arr_delay
    }])

    prediction = pipeline.predict(input_data)[0]
    result = "✅ Satisfied" if prediction == 1 else "❌ Not Satisfied"
    st.success(f"Prediction: {result}")
