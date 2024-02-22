import streamlit as st
import pandas as pd
from joblib import Parallel, delayed
import joblib

st.header('Diabetes Diagnose Application')
st.image('https://www.fitterfly.com/blog/wp-content/uploads/2022/08/How-to-Reduce-Sugar-Level-in-Blood-Immediately.jpg')


# df = pd.read_csv('data/diabetes_prediction_dataset.csv')
# st.dataframe(df.head())


col1, col2, col3, col4 = st.columns(4)

with col1:
   st.write("**Gender**")
   gender = st.selectbox(
    'select the gender',
    ('Female', 'Male', 'Other'))


with col2:
   st.write("**Age**")
   age = st.number_input("Insert your age")



with col3:
   st.write("**BMI**")
   bmi = st.number_input("Insert your bmi")

with col4:
    st.write("**Smoking History**")
    smoking_history = st.selectbox('select smoking history?',
   #  list(df['smoking_history'].unique())
   ['never', 'No Info', 'current', 'former', 'ever', 'not current']
)


col5, col6, col7= st.columns(3)

with col5:
   st.write("**HbA1c_Level**")
   hba1c_level = st.number_input("Insert your HbA1c Level")


with col6:
   st.write("**Blood_Glucose_Level**")
   blood_glucose_level = st.number_input("Insert your blood glucose")



with col7:
   st.write("**Disease**")
   heart_disese = int(st.checkbox("Heart Disease"))
   hyper_tension = int(st.checkbox("Hyper tension"))

# st.write(
# gender,
# age,
# hyper_tension,
# heart_disese,
# smoking_history,
# bmi,
# hba1c_level,
# blood_glucose_level)

if st.button('Predict'):

   diabetes_model = joblib.load('data/model/diabetes_model.pkl')
   columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
      'bmi', 'HbA1c_level', 'blood_glucose_level']
   # y_pred = diabetes_model.predict(pd.DataFrame([[gender, age, hyper_tension, heart_disese, smoking_history, bmi, hba1c_level, blood_glucose_level]], columns=df.columns[0:-1]))
   y_pred = diabetes_model.predict(pd.DataFrame([[gender, age, hyper_tension, heart_disese, smoking_history, bmi, hba1c_level, blood_glucose_level]], columns=columns))

   st.markdown('## Result')

   if y_pred[0] == 0:

      st.success('No Diabetes')

   else:
      st.error('Has Diabetes')

