import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
with open("model_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.title("✈️ Flight Passenger Satisfaction Predictor")

# Input Fields
gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
age = st.number_input("Age", min_value=7, max_value=100)
type_of_travel = st.selectbox("Type of Travel", ["Personal Travel", "Business Travel"])
travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
flight_distance = st.number_input("Flight Distance", min_value=50, max_value=5000)

inflight_wifi_service = st.slider("Inflight wifi service", 0, 5)
online_boarding = st.slider("Online boarding", 0, 5)

# Add other required service-related sliders (must match training columns)
departure_arrival_time_convenient = st.slider("Departure/Arrival time convenient", 0, 5)
ease_of_online_booking = st.slider("Ease of Online booking", 0, 5)
food_and_drink = st.slider("Food and drink", 0, 5)
seat_comfort = st.slider("Seat comfort", 0, 5)
inflight_entertainment = st.slider("Inflight entertainment", 0, 5)
on_board_service = st.slider("On-board service", 0, 5)
leg_room_service = st.slider("Leg room service", 0, 5)
baggage_handling = st.slider("Baggage handling", 0, 5)
checkin_service = st.slider("Checkin service", 0, 5)
inflight_service = st.slider("Inflight service", 0, 5)
cleanliness = st.slider("Cleanliness", 0, 5)
departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, max_value=1000)
arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, max_value=1000)

# Prediction
if st.button("Predict Satisfaction"):
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": type_of_travel,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Inflight wifi service": inflight_wifi_service,
        "Departure/Arrival time convenient": departure_arrival_time_convenient,
        "Ease of Online booking": ease_of_online_booking,
        "Food and drink": food_and_drink,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "Online boarding": online_boarding,
        "Leg room service": leg_room_service,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "On-board service": on_board_service,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay
    }])

    prediction = pipeline.predict(input_data)[0]
    result = "✅ Satisfied" if prediction == 1 else "❌ Not Satisfied"
    st.success(f"Prediction: {result}")
