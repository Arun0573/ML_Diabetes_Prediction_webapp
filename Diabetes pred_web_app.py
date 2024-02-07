# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:48:08 2024

@author: arunk
"""

import numpy as np
import pickle  
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('D:/ML_project/Diabetes Prediction/trained_model.sav', 'rb'))

# Create a function for prediction
def diabetes_prediction(input_data):
    
    # Changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshaping
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Making prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')
    
    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    Bloodpressure = st.text_input('BloodPressure value')
    Skinthickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree function value')
    Age = st.text_input('Age of the Person')
    
    # Code for prediction
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, Bloodpressure, Skinthickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
