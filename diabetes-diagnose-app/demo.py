import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import joblib
from helper_function import load_df


st.header('Diabetes Diagnose Application')
st.image('https://www.fitterfly.com/blog/wp-content/uploads/2022/08/How-to-Reduce-Sugar-Level-in-Blood-Immediately.jpg')

df = load_df('/Users/andishetavakkoli/Documents/notebook/github_project/diabetes-diagnose-app/data/diabetes_prediction_dataset.csv')

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write('**Gender**')
    gender = st.selectbox('select a gender from box',
                          ('Female', 'Male', 'Other'))

with col2:
    st.write('**Age**')
    age = st.number_input('select an age')



with col3:
    st.write('**BMI**')
    bmi = st.number_input('select a bmi number')


with col4:
    st.write('**Smoking History**')
    smoking_history = st.selectbox('select a smoking history from box',
                        (df['smoking_history'].unique()))


col5, col6, col7 = st.columns(3)
with col5:
    st.write('**HbA1c_Level**')
    hba1c_level = st.number_input('select a HbA1c Level number')

with col6:
    st.write('**Blood_Glucose_Level**')
    blood_glucose = st.number_input('select a blood glucose Level number')

with col7:
    st.write('**Diseas**')
    heart_disease = st.checkbox('Heart Diseas')
    if heart_disease:
        st.write('Yes')

    hypertension = st.checkbox('Hyper Tension')
    if hypertension:
        st.write('Yes')
        
if st.button('Predict'):
    # load preprocessore and model
    preprocessor = joblib.load('/Users/andishetavakkoli/Documents/notebook/github_project/diabetes-diagnose-app/data/model/preprocessor.joblib')
    loaded_model = joblib.load('/Users/andishetavakkoli/Documents/notebook/github_project/diabetes-diagnose-app/data/model/diabetes_model.joblib')

    df_sample = pd.DataFrame([[gender , age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose]], 
                                columns=['gender',
                                            'age',
                                            'hypertension',
                                            'heart_disease',
                                            'smoking_history',
                                            'bmi',
                                            'HbA1c_level',
                                            'blood_glucose_level'
                                            ])


    df_sample = preprocessor.transform(df_sample)
    result = model.predict(df_sample)[0]

    if result == 1:
        st.success('You have no diabetes!', icon="✅")
    else:
        st.error('You have diabetes', icon="🚨")
    


    



