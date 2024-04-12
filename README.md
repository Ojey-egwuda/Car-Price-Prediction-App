# Car-Price-Prediction-App
Welcome to the Car Price Prediction App! This web application predicts the price of used cars based on various features such as brand, model, year, mileage, and condition. The app empowers users to make informed decisions when buying or selling used cars by providing reliable price estimates based on data-driven insights.

# Overview
The Car Price Prediction App is a user-friendly tool built using Streamlit, a Python library for creating interactive web applications. It leverages a machine learning model trained on historical car data (2000 - 2024) scraped from a Nigerian car dealership website, Car45.com to provide accurate price estimates.

# Features
Predict Car Price: Input the car's brand, model, condition, year, and mileage to get an instant price prediction.  
Interactive Interface: User-friendly interface with dropdown menus and sliders for easy input.  
Price Range: Provides a price range indicating the potential variation in price based on overall condition and mechanical issues.

# How it Works
1. User Input: Users provide details about the car they want to evaluate, including brand, model, condition (Nigerian Used or Foreign Used), year of manufacture, and mileage.
   
2. Processing Input: The app processes the user input by encoding categorical variables and scaling numerical features to prepare them for prediction.

3. Model Prediction: The processed input is passed to a pre-trained XGBoost regression model. This model was trained on historical car listing data obtained from Car45.com.

4. Price Prediction: The regression model predicts a single price based on the input features provided by the user.

5. Display Results: The predicted price is displayed to the user, along with a price range indicating potential variations in price based on overall condition and any mechanical issues.

# Model Training
The XGBoost model used in the app was trained using the historical car data from Car45.com. The training process involved fitting the model to the dataset using Scikit-Learn. After training, the model was serialized using Pickle and integrated into the Streamlit app for real-time predictions.
