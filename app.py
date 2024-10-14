# Import necessary libraries
import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the XGBoost model
pickle_in = open("xgboost_model.pkl", "rb")
xgb_model = pickle.load(pickle_in)

# Define a preprocessing function to handle boolean and other necessary transformations
def preprocess_data(df):
    # List of all required columns for the model
    required_columns = [
        'city', 'state', 'latitude', 'longitude', 'stars', 'review_count',
        'is_open', 'BusinessAcceptsCreditCards', 'RestaurantsPriceRange2',
        'Restaurants', 'Food', 'Shopping', 'Home Services', 'Beauty & Spas',
        'Nightlife', 'Health & Medical', 'Local Services', 'Bars', 'Automotive',
        'garage', 'street', 'lot', 'valet'
    ]
    
    # Add any missing columns and fill with default values (e.g., 0 for numerical columns)
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns with 0
    
    # Convert boolean-like columns to 0/1
    boolean_columns = ['garage', 'street', 'lot']
    df[boolean_columns] = df[boolean_columns].astype(int)

    # Fill missing values for numeric columns (if any)
    df = df.fillna(0)
    
    # Ensure only the required columns are passed
    df = df[required_columns]
    
    return df

# Define the prediction function
def predict_parking_validated(input_data):
    """
    Predict if a business parking is validated or not.
    """
    predictions = xgb_model.predict(input_data)
    return predictions

# Streamlit application
def main():
    st.title("Business Parking Validation Prediction")

    # HTML formatting for app appearance
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Parking Validator ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Preprocess the data
        try:
            df_preprocessed = preprocess_data(df)
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            return

        # Display prediction button and process
        if st.button("Predict"):
            predictions = predict_parking_validated(df_preprocessed)
            
            # Convert the numerical predictions into human-readable labels
            validated_results = ["Parking Validated" if pred == 1 else "Parking Not Validated" for pred in predictions]
            
            # Display results for each business in the CSV
            for i, result in enumerate(validated_results):
                st.success(f"Business {i+1}: {result}")
    
    # Display about information
    if st.button("About"):
        st.text("This app predicts if a business's parking is validated based on inputs like business stars, review stars, etc.")
        st.text("Built with Streamlit and XGBoost")

# Run the Streamlit app
if __name__ == '__main__':
    main()
