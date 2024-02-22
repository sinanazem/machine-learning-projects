from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import LinearSVC
import pandas as pd
from pathlib import Path
from sentiment_analyzer import sentiment_analysis
from data_preprocessing import text_preprocessing
from loguru import logger
import pickle
from joblib import Parallel, delayed
import joblib

p= Path('./data')
all_data_files = list(p.glob('**/*.csv'))



def read_df():
    """read and concate and process two dataframes"""

    list_df = [pd.read_csv(path) for path in all_data_files]

    df = pd.concat(list_df , axis=0, ignore_index=True)
    df = df.dropna()
    df['Description'] = df['Description'].apply(text_preprocessing)
    df = sentiment_analysis(df, 'Description')

    return df




def build_model(df):

    X = df['Description']
    y = df['ds_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('svc', LinearSVC())])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # logger.info(f'train model score: {pipeline.score(X_train, y_train)}')
    # logger.info(f'test model score: {pipeline.score(X_test, y_test)}')
    # logger.info(f'accuracy score: {accuracy_score(y_test, y_pred)}')

    return pipeline


if __name__ == '__main__':

    df = read_df()
    pipeline_model = build_model(df)
    sent1 = ['GST officers detect Rs 4,000 crore of ITC fraud in April-June']
    sent2 = ["Finance Ministry releases Rs 9,871 crore to 17 states as grant"]
    print(pipeline_model.predict(sent1))
    print(pipeline_model.predict(sent2))




    # Save the model as a pickle in a file
    joblib.dump(pipeline_model, 'finance_news_model.pkl')




