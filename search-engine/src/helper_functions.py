#  Read Vanguard Dataframe
from utils import DataBaseClass


def read_data():
    """ This function read data from database"""

    obj_data_base_class = DataBaseClass('guest', 'Aa12345', 'localhost', 5432, 'insights_db')

    vanguard_command = """select * from insights_data"""

    vanguard_data = obj_data_base_class.db_query(
        command=vanguard_command,
        read_query=True
    )
    return vanguard_data

def read_columns():
    """ This function read columns from database"""
    
    obj_data_base_class = DataBaseClass('guest', 'Aa12345', 'localhost', 5432, 'insights_db')
    vanguard_col_command = """select column_name from information_schema.columns where table_name='insights_data'"""

    vanguard_col = obj_data_base_class.db_query(
        
        command=vanguard_col_command,
        read_query=True
    )
    vanguard_col = [tup[0] for tup in vanguard_col]
    return vanguard_col
