import io # standard library for input and output

import psycopg2  # this library needs to connect to db with python
from loguru import logger
from sqlalchemy import create_engine # it is connects to db with python


class DataBaseClass:

    """this db get some inputs such as username, password, host,... 
    and through this we can connect to database, read, insert, create, edit and ect
    
    
    """

    def __init__(self, username, password, host, port, db_name):

        # for connect to db needs bellow informations
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name

    def db_query(self, command, read_query=None):
        """ run query in PostgreSQL data """

        try:

            with psycopg2.connect(database=self.db_name,
                                  user=self.username,
                                  password=self.password,
                                  host=self.host,
                                  port=self.port) as conn:

                with conn.cursor() as curs:
                    conn.autocommit = True
                    curs.execute(command)
                    if read_query:
                        data = curs.fetchall()

            if read_query:
                return data

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def create_database(self, command):
        """ create db """
        self.db_query(command=command)
        logger.info(f'database is created.')

    def create_tables(self, command):
        """ create tables in the PostgreSQL database"""
        self.db_query(command=command)
        logger.info(f'table is created.')

    def insert_table(self, command):
        """ insert a new data in the table """

        self.db_query(command=command)
        logger.info(f'insert into table is succefully!.')

    def read_table(self, command):

        try:

            logger.info(f'Reading From table...')
            return self.db_query(command=command, read_query=True)

        except Exception as e:
            print(e)

    def store_data(self, df, sql_table_name):

        """ connect to db with sqlalchemy , it gets a pandas dataframe and put it in db"""

        engine = create_engine(
            f'postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}')

        df.to_sql(sql_table_name, engine, if_exists='append',
                          index=False)  # drops old table and creates new empty table

        # df.head(0).to_sql(sql_table_name, engine, if_exists='replace',
        #                   index=False)  # drops old table and creates new empty table

        # conn = engine.raw_connection()
        # logger.info(f'You are connected to the database!')
        # cur = conn.cursor()
        # output = io.StringIO()
        # df.to_csv(output, sep='\t', header=False, index=False)
        # output.seek(0)
        # contents = output.getvalue()
        # cur.copy_from(output, sql_table_name, null="")  # null values become ''
        # conn.commit()
        # logger.info(f'The data was successfully stored in the database.')