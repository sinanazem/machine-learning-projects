import pandas as pd
from loguru import logger

def load_df(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(df.head(2))
        logger.info('Dataframe read!')
        return df

    except Exception as e:
        print(e)