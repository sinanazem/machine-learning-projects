import streamlit as st
from model import build_model, read_df
from joblib import Parallel, delayed
import joblib



st.header('Sentiment Analysis of Financial News')
st.image('https://www.surveysensum.com/wp-content/uploads/2020/02/SENTIMENT-09-1.png')

query = st.text_input('Ask Query:')


if __name__ == '__main__':
    # df = read_df()
    # pipeline_model = build_model(df)

    # Load the model from the file
    model_from_joblib = joblib.load('finance_news_model.pkl')

    # Use the loaded model to make predictions
    result = model_from_joblib.predict([query])

    st.write(result)