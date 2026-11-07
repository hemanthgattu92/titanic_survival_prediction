import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the models (make sure these files are in the same directory as app.py)
model = joblib.load('svm_titanic.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('colums.pkl')

# 2. Design the Streamlit UI
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict survival.")

# Input fields matching the model columns
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0, max_value=600, value=32)
embarked = st.selectbox("Port of Embarkation (0=C, 1=Q, 2=S)", [0, 1, 2])
alone = st.selectbox("Travelling Alone (0=No, 1=Yes)", [0, 1])

# 3. Handle prediction on button click
if st.button("Predict"):
    # Map inputs
    sex_encoded = 1 if sex == "Male" else 0
    
    # Prepare the input DataFrame exactly how the model expects
    user_features = pd.DataFrame([[
        pclass, sex_encoded, age, sibsp, parch, fare, embarked, alone
    ]], columns=columns)
    
    # Scale features
    scaled_features = scaler.transform(user_features)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    
    if prediction == 1:
        st.success("🎉 The passenger is predicted to have SURVIVED.")
    else:
        st.error("☠️ The passenger is predicted to have NOT SURVIVED.")
