import pandas as pd
import sqlalchemy
import os
from os.path import join, dirname
from dotenv import load_dotenv
import numpy as np


dotenv_path = join(dirname(__file__),'.env')
load_dotenv(dotenv_path)
DATABASE_URI = os.environ.get("DATABASE_URI")

def pandas_to_sql(dF):
    print('here')
    db = sqlalchemy.create_engine(DATABASE_URI)
    dF.to_sql('brs',db)
    print('brs')

def sql_to_pandas(query):
    db = sqlalchemy.create_engine(DATABASE_URI)
    dF = pd.read_sql(query, db)
    return dF



if __name__ == "__main__":
    dF = pd.read_csv("data/data.csv")

    if False:
    #dF.columns = map(str.lower, dF.columns)
        print(dF.head(),len(dF))
        pandas_to_sql(dF)
    query = "SELECT * FROM brs WHERE index IN ( SELECT MIN(index) FROM brs GROUP BY \"beer_beerId\", \"user_profileName\")"
    print(query)
    #dF2 = sql_to_pandas("SELECT index, review_overall, user_profilename  FROM brss")

    dF2 = sql_to_pandas(query)
#    dF.columns = map(str.lower, dF.columns)

    print(dF2.keys())
    print(len(dF2),len(dF[['beer_beerId','user_profileName']]),len(dF[['beer_beerId','user_profileName']].drop_duplicates()))
    print(np.all(np.sort(dF2['beer_beerId'].values) == np.sort(dF[['beer_beerId','user_profileName']].drop_duplicates()['beer_beerId'].values)))


#    print(len(dF),len(dF2))

    #print(dF.head(),dF2.head())
    #for r in result_set:
    #  print(r)
