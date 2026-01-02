# Car Price Prediction App
### Run the application from this link 

This repository hosts a Streamlit application designed for predicting used car selling prices based on various vehicle attributes. The core of this application is a machine learning model trained on a comprehensive dataset of used cars.
🚀 Project Overview

This project focuses on building and deploying a car price prediction model. The data underwent rigorous cleaning and preprocessing, including handling missing values, extracting numerical features from text (e.g., mileage, engine, max_power, torque), and encoding categorical variables (brand, fuel, transmission, owner, seller_type).

An Extra Trees Regressor model was selected for its robust performance in capturing complex non-linear relationships within the data. The target variable, selling_price, was log-transformed (np.log1p) during training to address skewness and improve model performance, with predictions inverse-transformed (np.expm1) for user interpretability.
✨ Key Features

    Interactive UI: A user-friendly interface built with Streamlit for real-time price predictions.
    Robust Preprocessing: Includes steps for handling missing data, converting data types, and feature engineering (e.g., extracting car brand from name).
    Machine Learning Model: Utilizes a pre-trained ExtraTreesRegressor for accurate predictions.
    Scalability: Numerical features are standardized using StandardScaler to ensure optimal model performance.
    Categorical Encoding: Comprehensive mappings for categorical features are handled internally to transform user inputs into model-ready numerical representations.

🛠️ Technologies Used

    Python 3.x
    pandas for data manipulation
    numpy for numerical operations
    scikit-learn for preprocessing (e.g., LabelEncoder, StandardScaler) and the machine learning model (ExtraTreesRegressor)
    streamlit for building the interactive web application
    joblib for model and scaler persistence


<img width="1753" height="874" alt="image" src="https://github.com/user-attachments/assets/f6f0d9f3-c405-47ad-b3a8-0766dcd7681c" />
