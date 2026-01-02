import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained ExtraTreesRegressor model and the numerical scaler
# Ensure these paths are correct in your Streamlit deployment environment
model = joblib.load("dtrcarpricepredict.pkl")
scaler = joblib.load("num_scaler.joblib")

# --- Define categorical feature mappings (reconstructed from notebook state) ---
mappings = {
    "brand": {
        "classes": ['Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'],
        "encoded_values": list(range(32))
    },
    "fuel": {
        "classes": ['CNG', 'Diesel', 'LPG', 'Petrol'],
        "encoded_values": list(range(4))
    },
    "transmission": {
        "classes": ['Automatic', 'Manual'],
        "encoded_values": list(range(2))
    },
    "owner": {
        "classes": ['First Owner', 'Fourth & Above Owner', 'Second Owner', 'Test Drive Car', 'Third Owner'],
        "encoded_values": list(range(5))
    },
    "seller_type": {
        "classes": ['Dealer', 'Individual', 'Trustmark Dealer'],
        "encoded_values": list(range(3))
    }
}

# Helper function to get encoded value from original value using mappings
def get_encoded_value(col_name, original_value, mappings_dict):
    if col_name in mappings_dict:
        class_list = mappings_dict[col_name]["classes"]
        encoded_list = mappings_dict[col_name]["encoded_values"]
        try:
            index = class_list.index(original_value)
            return encoded_list[index]
        except ValueError:
            return None  # Handle unseen categories
    return None

# Define the prediction function for Streamlit
def predict_car_price(brand, year, km_driven, fuel, transmission, engine, max_power, mileage, seats, torque, owner, seller_type):
    # Encode categorical features
    brand_encoded = get_encoded_value('brand', brand, mappings)
    fuel_encoded = get_encoded_value('fuel', fuel, mappings)
    transmission_encoded = get_encoded_value('transmission', transmission, mappings)
    owner_encoded = get_encoded_value('owner', owner, mappings)
    seller_type_encoded = get_encoded_value('seller_type', seller_type, mappings)

    # Check if any encoding failed (e.g., unseen category)
    if None in [brand_encoded, fuel_encoded, transmission_encoded, owner_encoded, seller_type_encoded]:
        st.error("Error: Could not encode one or more categorical features. Please ensure valid selections.")
        return None

    # Create a DataFrame for numerical features to be scaled
    numerical_input_df = pd.DataFrame([[year, km_driven, engine, max_power, mileage, seats, torque]],
                                      columns=['year', 'km_driven', 'engine', 'max_power', 'mileage', 'seats', 'torque'])

    # Scale numerical features using the loaded scaler
    scaled_numerical_input = scaler.transform(numerical_input_df)

    # Combine all features into a single DataFrame for prediction
    feature_order = ['brand_encoded', 'year', 'km_driven', 'fuel_encoded', 'transmission_encoded',
                     'engine', 'max_power', 'mileage', 'seats', 'torque', 'owner_encoded', 'seller_type_encoded']

    input_data = {
        'brand_encoded': brand_encoded,
        'year': scaled_numerical_input[0, 0],
        'km_driven': scaled_numerical_input[0, 1],
        'fuel_encoded': fuel_encoded,
        'transmission_encoded': transmission_encoded,
        'engine': scaled_numerical_input[0, 2],
        'max_power': scaled_numerical_input[0, 3],
        'mileage': scaled_numerical_input[0, 4],
        'seats': scaled_numerical_input[0, 5],
        'torque': scaled_numerical_input[0, 6],
        'owner_encoded': owner_encoded,
        'seller_type_encoded': seller_type_encoded
    }

    final_input_df = pd.DataFrame([input_data], columns=feature_order)

    # Make prediction (log-transformed price)
    log_price_prediction = model.predict(final_input_df)[0]

    # Inverse transform the prediction to get actual price
    predicted_price = np.expm1(log_price_prediction)

    return predicted_price

# --- Streamlit UI ---
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")
st.title("🚗 Used Car Price Prediction")
st.markdown("Enter the details of the car to get an estimated selling price.")

with st.form("car_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("Brand", options=mappings['brand']['classes'])
        year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, value=2015, step=1)
        km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)
        fuel = st.selectbox("Fuel Type", options=mappings['fuel']['classes'])

    with col2:
        transmission = st.selectbox("Transmission Type", options=mappings['transmission']['classes'])
        engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200, step=10)
        max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, value=80.0, step=0.1)
        mileage = st.number_input("Mileage (kmpl)", min_value=1.0, max_value=50.0, value=18.0, step=0.1)

    with col3:
        seats = st.number_input("Seats", min_value=1, max_value=10, value=5, step=1)
        torque = st.number_input("Torque (Nm)", min_value=50.0, max_value=1000.0, value=150.0, step=0.1)
        owner = st.selectbox("Owner Type", options=mappings['owner']['classes'])
        seller_type = st.selectbox("Seller Type", options=mappings['seller_type']['classes'])

    submitted = st.form_submit_button("Predict Price")

    if submitted:
        with st.spinner('Calculating predicted price...'):
            predicted_price = predict_car_price(brand, year, km_driven, fuel, transmission,
                                                 engine, max_power, mileage, seats, torque, owner, seller_type)
            if predicted_price is not None:
                st.success(f"### Estimated Selling Price: ₹{predicted_price:,.2f}")
                st.balloons()
