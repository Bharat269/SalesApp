import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Load the model
@st.cache_resource
def load_model():
    with open('SalesModel.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_data(data):
    # Convert date to datetime if it's a string
    if isinstance(data['date'].iloc[0], str):
        data['date'] = pd.to_datetime(data['date'])
    
    # Extract time features
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    
    # Create lag features
    data['lag_1'] = data['quantity'].shift(1)
    data['lag_3'] = data['quantity'].shift(3)
    
    # Drop rows with NaN values
    data = data.dropna()
    
    return data[['year', 'month', 'day', 'lag_1', 'lag_3']]

def main():
    st.title("Sales Prediction App")
    
    # Load the model
    model = load_model()
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])
    
    with tab1:
        st.header("Manual Input")
        
        # Date input
        date = st.date_input("Select Date", datetime.today())
        
        # Previous sales inputs
        quantity_1 = st.number_input("Sales from Previous Month (lag_1)", min_value=0.0)
        quantity_3 = st.number_input("Sales from 3 Months Ago (lag_3)", min_value=0.0)
        
        if st.button("Predict"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'year': [date.year],
                'month': [date.month],
                'day': [date.day],
                'lag_1': [quantity_1],
                'lag_3': [quantity_3]
            })
            
            # Make prediction
            prediction = model.predict(input_data)
            
            st.success(f"Predicted Sales: {prediction[0]:.2f}")
    
    with tab2:
        st.header("CSV Upload")
        
        # File upload
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read CSV
                data = pd.read_csv(uploaded_file)
                
                # Show raw data
                st.subheader("Raw Data Preview")
                st.write(data.head())
                
                # Check if required columns exist
                required_cols = ['date', 'quantity']
                if not all(col in data.columns for col in required_cols):
                    st.error("CSV must contain 'date' and 'quantity' columns!")
                    return
                
                # Preprocess data
                X = preprocess_data(data)
                
                # Make predictions
                predictions = model.predict(X)
                
                # Add predictions to original data
                results = data.copy()
                results['predicted_quantity'] = predictions
                
                # Show results
                st.subheader("Predictions")
                st.write(results)
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="sales_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()