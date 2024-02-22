import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from helper_functions import read_columns, read_data


    #export
def preprocess(title, body=None):
    """ Preprocess the input, i.e. lowercase, remove html tags, special character and digits."""
    text = ''
    if body is None:
        text = title
    else:
        text = title + body
    # to lower case
    text = text.lower()

    # remove tags
    text = re.sub("</?.*?>"," <> ", text)
    
    # remove special characters and digits
    text = re.sub("(\\d|\\W)+"," ", text).strip()
    return text
    
def create_tfidf_features(corpus, max_features=5000, max_df=0.95, min_df=2):
    """ Creates a tf-idf matrix for the `corpus` using sklearn. """
    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word', 
                                    stop_words='english', ngram_range=(1, 1), max_features=max_features, 
                                    norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                    max_df=max_df, min_df=min_df)
    X = tfidf_vectorizor.fit_transform(corpus)
    print('tfidf matrix successfully created.')
    return X, tfidf_vectorizor

def calculate_similarity(X, vectorizor, query, top_k=5):
    """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of 
    the `query` and `X` (all the documents) and returns the `top_k` similar documents."""
    
    # Vectorize the query to the same length as documents
    query_vec = vectorizor.transform(query)
    # Compute the cosine similarity between query_vec and all the documents
    cosine_similarities = cosine_similarity(X,query_vec).flatten()
    # Sort the similar documents from the most similar to less similar and return the indices
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
    return (most_similar_doc_indices, cosine_similarities)


def most_similar_article(query):

    company_insight_df = pd.DataFrame(read_data(), columns=read_columns())

    # Preprocess the corpus
    data = [preprocess(title, body) for title, body in zip(company_insight_df['title'], company_insight_df['content'])] 

   
    # Learn vocabulary and idf, return term-document matrix
    X,v = create_tfidf_features(data)
    
    bow_df = pd.DataFrame(X.toarray())
    final_df = pd.concat([bow_df, company_insight_df], axis=1)

    user_question = [query]
    sim_vecs, cosine_similarities = calculate_similarity(X, v, user_question)
    
    final_df = final_df.iloc[sim_vecs]
    final_df = final_df[['company', 'topic', 'title', 'date', 'link', 'content']]
    return final_df
