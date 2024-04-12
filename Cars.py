import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import time
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('final_xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# Get the feature names from the trained model
model_feature_names = model.get_booster().feature_names

# Title and Page Configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'> Used Car Price Prediction </h1>", unsafe_allow_html=True)
image = Image.open('cars.jpg')
st.image(image, use_column_width=True)


st.markdown("<h2 style=>Contact the Developer</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px;'>Connect with Egwuda Ojonugwa on : "
            "<a href='https://www.linkedin.com/in/egwudaojonugwa/' style='color: #00CED1;'>LinkedIn</a></p>",
            unsafe_allow_html=True)


# Car brand and model dictionary
car_brands = ['Toyota', 'Ford', 'Honda', 'Mercedes-Benz', 'Mazda', 'Infiniti', 'Nissan', 'Acura', 'Land Rover', 'Hyundai', 'Lexus', 'Peugeot', 'Pontiac', 'Chevrolet', 'Kia', 'Lincoln', 'Volkswagen', 'Mitsubishi', 'Suzuki', 'BMW', 'Genesis', 'GAC', 'Subaru', 'Scion', 'Volvo', 'Audi', 'Chrysler', 'Cadillac', 'MG', 'Sinotruk', 'Porsche', 'Jaguar', 'Renault', 'Opel', 'Dodge', 'Skoda', 'Jeep', 'GMC', 'Changan', 'Foton', 'Saturn', 'Buick']

car_models = {
    'Toyota': ['4-Runner', 'Avalon', 'Avensis', 'Avensis Verso', 'C-HR', 'Camry', 'Corolla', 'Estima', 'Fortuner', 'Hiace', 'Highlander', 'Highlander Hybrid', 'Highlander Limited',  'Hilux', 'Land Cruiser', 'Land Cruiser Prado', 'Matrix', 'Picnic', 'Prius', 'Prius HSD Hybrid', 'RAV4', 'Sequoia', 'Solara', 'Sienna', 'Sienna CE', 'Sienna LE', 'Sienna XLE', 'Tacoma', 'Tundra', 'Venza', 'Venza XLE', 'Vios', 'Yaris',],
    'Ford': ['E-150', 'E-250', 'Edge', 'Edge SE', 'Escape', 'Expedition', 'Explorer', 'F-150', 'Focus', 'Freestar', 'Fusion',  'Maverick', 'Mondeo', 'Mustang', 'Ranger', 'Taurus', 'Transit',],
    'Honda': ['Accord', 'City', 'Civic', 'CR-V', 'Crosstour', 'Element', 'HR-V', 'Insight EX', 'Odyssey', 'Pilot', 'Ridgeline',],
    'Mercedes-Benz': ['A 180', 'B 200', 'C 180', 'C 200', 'C 230', 'C 240', 'C 250', 'C 280', 'C 300', 'C 320', 'C 350', 'C 63', 'CLA 250', 'CLK', 'CLS 350', 'CLS 550', 'E 200', 'E 250', 'E 300', 'E 320', 'E 350', 'E 400', 'E 500', 'E 550', 'E 63', 'G 500', 'G 55 AMG', 'GL 450', 'GL 450 4MATIC', 'GL 550', 'GL 550 4MATIC', 'GLA 250', 'GLC 250 4MATIC', 'GLC 300', 'GLE 350', 'GLE 350 4MATIC', 'GLE 53 AMG', 'MB100', 'ML 320', 'ML 350', 'ML 350 4MATIC', 'R-Class', 'S 500', 'S 550', 'S 550 4MATIC', 'S 63 AMG', 'Sprinter', 'Vaneo',],
    'Mazda': ['3', '5', '6', '626', 'CX-9', 'CX-7', 'MPV' ,'MX-5', 'Premacy'],
    'Infiniti': ['FX35', 'G35', 'M35', 'QX56', 'QX60', 'QX4', 'JX35'],
    'Nissan': ['Almera', 'Altima', 'Armada', 'Frontier', 'Juke', 'Micra', 'Murano', 'Maxima', 'Navara', 'Pathfinder', 'Primera', 'Qashqai', 'Quest', 'Rogue', 'Sentra', 'Sunny', 'Tiida', 'Titan', 'Versa', 'X-Trail', 'Xterra',],
    'Acura': ['MDX', 'RDX', 'TL','TLX', 'TSX', 'RL', 'ZDX'],
    'Land Rover': ['LR2', 'LR3', 'LR4', 'Land Rover', 'Range Rover', 'Range Rover Sport', 'Rover Discovery', 'Range Rover Evoque', 'Range Rover Velar', 'Range Rover Vogue',],
    'Hyundai': ['Accent', 'Azera', 'Creta', 'Elantra', 'Entourage', 'Genesis', 'Grandeur', 'Palisade', 'Ix35', 'Palisade', 'Santa Fe', 'Sonata', 'Tucson', 'Veloster', 'Veracruz', 'XG350', ],
    'Lexus': ['ES 300', 'ES 330', 'ES 350', 'GS 300', 'GS 350', 'GS 430', 'GS 460', 'GX 450', 'GX 460', 'GX 470', 'IS 250', 'IS 350', 'LS 460', 'LX 570', 'NX 350', 'RX 300', 'RX 330', 'RX 350', 'RX 400h', 'RX 450h',],
    'Peugeot': ['206', '207 Sport', '307', '308', '406', '407', '508',  '607', 'Partner', 'RC',],
    'Pontiac': ['Torrent', 'Vibe',],
    'Chevrolet': ['Avalanche', 'Blazer', 'Captiva', 'Cruze', 'Epica', 'Express', 'Equinox', 'Impala', 'Lacetti', 'Malibu', 'Nubira', 'Orlando', 'Suburban', ],
    'Kia': ['Carnival', 'Cerato', 'Mohave', 'Optima', 'Rio', 'Sedona', 'Sorento', 'Soul', 'Sportage',],
    'Lincoln': ['MKX',],
    'Volkswagen': ['Bora', 'Comfort Coupe', 'Caddy', 'Golf', 'Jetta', 'Passat', 'Polo', 'Routan', 'Sharan', 'Touareg', 'Touran',],
    'Mitsubishi': ['Carisma', 'Grandis', 'L200', 'Lancer', 'Montero', 'Outlander',  'Pajero', 'Spacestar', ],
    'Suzuki': ['XL-7', 'Grand Vitara',],
    'BMW': ['3 Series', '318i', '320i', '325i', '328i', '450', '5 Series', '525i', '528i', '535i', 'X3', 'X5', 'X6',],
    'Genesis': ['G80',],
    'GAC': ['GA4', 'GS4',],
    'Subaru': ['Legacy',  'Outback', 'Tribeca',],
    'Scion': ['xB', 'iA Base',],
    'Volvo': ['S40', 'S60', 'S80', 'XC90',],
    'Audi': ['A4', 'A6',],
    'Chrysler': ['300', 'Pacifica', 'Town & Country',],
    'Cadillac': ['SRX',],
    'MG': ['Magnett',],
    'Sinotruk': ['Howo',],
    'Porsche': ['Cayenne', 'Macan',],
    'Jaguar': ['X-Type',],
    'Renault': ['Duster',],
    'Opel': ['Astra', 'Meriva', 'Vectra', 'Zafira',],
    'Dodge': ['Avenger', 'Caravan', 'Charger', 'Challenger', 'Dart', 'Journey',],
    'Skoda': ['Octavia', 'Superb',],
    'Jeep': ['Compass', 'Cherokee', 'Wrangler',],
    'GMC': ['Acadia', 'Terrain',],
    'Changan': ['CS35',],
    'Foton': ['Sup',],
    'Saturn': ['Vue',],
    'Buick': ['Enclave',]
}

st.sidebar.markdown("<h1 style='text-align: center;'>Get a Car Price</h1>", unsafe_allow_html=True)

# Get user input from the sidebar
car_brand = st.sidebar.selectbox("Car Brand", sorted(car_models.keys()), key="car_brand_select")
car_model = st.sidebar.selectbox("Car Model", car_models[car_brand], key="car_model_select")
car_condition = st.sidebar.selectbox("Car Condition", ['Nigerian Used', 'Foreign Used'], key="car_condition_select")
year = st.sidebar.number_input("Year", min_value=2000, max_value=2024, key="year_input")
mileage = st.sidebar.number_input("Mileage", min_value=0, key="mileage_input")

# Define the function to make predictions
def predict_price(year, mileage, car_brand, car_model, car_condition, scaler):
    scaled_features = scaler.transform(np.array([[year, mileage]]))

    data = pd.DataFrame({
        'year': scaled_features[:, 0],
        'mileage': scaled_features[:, 1],
        'car_brand': [car_brand],
        'car_model': [car_model],
        'car_condition_encoded': [1 if car_condition == 'Nigerian Used' else 2]
    })

    data = pd.get_dummies(data, columns=['car_brand', 'car_model'], drop_first=False)

    data = data.reindex(columns=model_feature_names, fill_value=0)

    # Make the prediction
    predicted_log_price = model.predict(data)
    predicted_price = np.exp(predicted_log_price)  # Apply inverse logarithmic transformation

    return predicted_price[0]

predict_button = st.sidebar.button("Predict Price")

if predict_button:
    with st.spinner('Predicting...'):
        time.sleep(2)

        if car_brand and car_model:
            predicted_price = predict_price(year, mileage, car_brand, car_model, car_condition, scaler)
            st.sidebar.subheader(f'The predicted price is ₦{predicted_price:,.2f}')

            # Calculate price range
            min_price = predicted_price - predicted_price * 0.1102
            max_price = predicted_price + predicted_price * 0.1102

            text = f'The predicted price assumes the car is in average condition for {year} {car_brand} {car_model}.\n'
            text += f'Actual price may vary between ₦{min_price:,.2f} to ₦{max_price:,.2f} depending on overall condition and any mechanical issues.'
            st.sidebar.success(text)
        else:
            st.sidebar.warning("Please select a car brand and model.")


