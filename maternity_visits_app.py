import streamlit as st
import numpy as np
import joblib

# Load the trained logistic regression model
logreg = joblib.load('logistic_regression_model.pkl')

# Define the Streamlit app
st.title('Prediction of Receiving LLT')

st.write("""
This application predicts the maternals who visited ANC and received or didn't receive LLT based on various predictors.
Please provide the input data to get a prediction.
""")

# Define categorical to numeric mappings
Marital_status_map = {'Married': 1,'Widowed': 2, 'Single': 3, 'Divorced': 4, 'Separated': 5}
Breast_Exam_map = {'Normal': 1, 'Abnormal': 2, }
Diabetes_map = {'No Diabetes': 1, 'Has diabetes': 2, 'Not done': 3}
TB_Screening_map = {'Presumed TB': 1, 'No signs': 2, 'On TB treatment': 3,  'Not done': 4}
HIV_results_map = {'Positive': 1, 'Negative': 2, 'Prev Negative': 3}


# Create input fields for the user to enter data
No_of_ANC_visits=st.slider("No of ANC visits", min_value=1, max_value=8, value=0)
Breast_Exam = st.selectbox("Breast Exam", list(Breast_Exam_map.keys()))
Age = st.slider("Age", min_value=0, max_value=100, value=0)
Marital_status= st.selectbox("Marital",list(Marital_status_map.keys()))
Weight = st.slider("Weight", min_value=0, max_value=100, value=0)
Gestation_in_weeks = st.slider("Gestation in weeks", min_value=0, max_value=100, value=0)
Diabetes = st.selectbox("Diabetes", list(Diabetes_map.keys()))
TB_Screening = st.selectbox("TB Screening", list(TB_Screening_map.keys()))
HIV_results = st.selectbox("HIV results", list(HIV_results_map.keys()))


# Convert categorical inputs to numeric using the mappings
Marital_status_numeric = Marital_status_map[Marital_status]
Breast_Exam_numeric = Breast_Exam_map[Breast_Exam]
Diabetes_numeric = Diabetes_map[Diabetes]
TB_Screening_numeric = TB_Screening_map[TB_Screening]
HIV_results_numeric = HIV_results_map[HIV_results]

# Assuming the correct order of features used during training
# Prepare input data for prediction
input_data = np.array([Age, Marital_status_numeric, Weight, Gestation_in_weeks, Breast_Exam_numeric, Diabetes_numeric, TB_Screening_numeric, HIV_results_numeric, No_of_ANC_visits]).reshape(1, -1)

# Make prediction and display results
if st.button('Predict'):
    prediction = logreg.predict(input_data)
    if prediction[0] == 1:
        st.write("Prediction: Yes")
    else:
        st.write("Prediction: No")
