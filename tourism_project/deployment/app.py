
import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------

MODEL_PATH = "tourism_project/deployment/best_model.pkl"

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("Wellness Tourism Package Prediction")

st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to purchase the Wellness Tourism Package."
)


# ---------------------------------------------------------
# Collect customer inputs
# ---------------------------------------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    max_value=60,
    value=10,
    step=1
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=3,
    step=1
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced", "Unmarried"]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=50,
    value=3,
    step=1
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

designation = st.selectbox(
    "Designation",
    ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    max_value=1000000,
    value=25000,
    step=1000
)


# ---------------------------------------------------------
# Create prediction DataFrame
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "DurationOfPitch": [duration_of_pitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "NumberOfFollowups": [number_of_followups],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income]
})


# ---------------------------------------------------------
# Display inputs and make prediction
# ---------------------------------------------------------

if st.button("Predict Purchase", type="primary"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The customer is likely to purchase the "
            "Wellness Tourism Package."
        )
    else:
        st.info(
            "The customer is unlikely to purchase the "
            "Wellness Tourism Package."
        )

    st.write(
        f"Purchase Probability: **{probability:.2%}**"
    )
