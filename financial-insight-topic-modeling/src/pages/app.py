import datetime
import re
import string

import nltk
import numpy as np
import pandas as pd
import spacy
import streamlit as st
from bertopic import BERTopic
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from transformers import pipeline
from umap import UMAP

kpmg_col1, kpmg_col2 = st.columns([1, 9])
with kpmg_col2:
    st.markdown("**| KPMG Innovation Project**")
with kpmg_col1:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/KPMG_logo.svg/1280px-KPMG_logo.svg.png',width=70)
st.header('Financial Insights Text Analysis')
st.image('https://nexocode.com/images/nlp-in-telecommunication.png')


st.markdown("### An overview of the data:")
df = pd.read_csv('/mnt/c/Users/user/OneDrive/Desktop/KPMG/innovation-project-Apr-2023/cp_articles 2.csv')
st.dataframe(df)
st.markdown("### Topic Modeling with BERTopic")


col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input('Start date', datetime.date(2022, 1, 2))

with col2:
    end_date = st.date_input('End date', datetime.date(2022, 1, 2))

selected_earnings_call = st.selectbox(
"Choose your Company Insights: ",
['All Insights', 'Capital Group', 'Vanguard', 'JPMorgan', 'Pimco'])
df_c = pd.read_csv("/mnt/c/Users/user/OneDrive/Desktop/KPMG/innovation-project-Apr-2023/content.csv")


def bert_model(series, return_df=False, return_obj=False):

    umap_obj = UMAP()
    bert_model = BERTopic(umap_model=umap_obj)
    topics, probability =  bert_model.fit_transform(series)
    docTopics_df = bert_model.get_document_info(series)
    if return_df:
        return docTopics_df
    if return_obj:
        return bert_model

obj = bert_model(df_c['preproces_content'], return_obj=True)
fig = obj.visualize_barchart()
st.plotly_chart(fig)
fig2 = obj.visualize_term_rank()
st.plotly_chart(fig2)
# read final data
# final_df = pd.read_csv('/mnt/c/Users/user/OneDrive/Desktop/KPMG/innovation-project-Apr-2023/final.csv')
# final_df['date'] = pd.to_datetime(final_df['date'])
# topics_over_time = obj.topics_over_time(final_df['bert_topic_name'].to_list(), final_df['date'].to_list())
# st.plotly_chart(topics_over_time)
