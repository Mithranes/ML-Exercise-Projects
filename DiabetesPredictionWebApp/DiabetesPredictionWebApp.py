import numpy as np
import pickle
import streamlit as st


# Loading the saved model
loaded_model = pickle.load(open('/DiabetesPredictionWebApp/trained_model.sav', 'rb'))



# Creating a function for prediction
def diabetes_prediction(input_data):
    input_data_as_array = np.asarray(input_data)

    # reshape → because model expects samples in 2D
    input_data_reshaped = input_data_as_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 1:
        return "The person is **Diabetic**"
    else:
        return "The person is **NOT Diabetic**"



def main():

    #Giving title
    st.title('Diabetes Prediction Web App')

    # Input fields
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0)
    Glucose = st.number_input('Glucose Level', min_value=0)
    BloodPressure = st.number_input('Blood Pressure value', min_value=0)
    SkinThickness = st.number_input('Skin Thickness value', min_value=0)
    Insulin = st.number_input('Insulin Level', min_value=0)
    BMI = st.number_input('BMI value', min_value=0.0)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0)
    Age = st.number_input('Age of the Person', min_value=0)

    # Code for Prediction
    diagnosis = ''

    # Creating a button for prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness,
                                         Insulin, BMI, DiabetesPedigreeFunction, Age])
    st.success(diagnosis)


if __name__ == '__main__':
    main()

