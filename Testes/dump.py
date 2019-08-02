import os
import json
import uuid
from sqlalchemy.engine import create_engine
from sqlalchemy import Column, Table, MetaData
from sqlalchemy import Integer, Text, Float
import sqlalchemy

import os
import pandas as pd
import seaborn as sns
import numpy as np  

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

from sklearn.model_selection import train_test_split


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

class Driver(object):

    def __init__(self):
        self.db = None
        self.engine = None
        self.meta = None

        self.address = 'postgresql+psycopg2://postgres:docker@localhost/postgres?port=5432'
        self.table_files_name = 'benchmarks'
        table_files_args = lambda: (
              Column('id', Text),
              Column('name', Text),
              Column('0,0', Float),
              Column('0,1', Float),
              Column('1,0', Float),
              Column('1,1', Float)
              )
        self.table_files = lambda x: Table(self.table_files_name, x, *table_files_args() )
        self.table_files_query = sqlalchemy.table(self.table_files_name, *table_files_args() )
        

    def connect(self):
        self.db = create_engine(self.address)
        self.engine = self.db.connect()
        self.meta = MetaData(self.engine)
        self.table_files(self.meta)
        self.meta.create_all()
        return self

    def insert(self, params):
        statement = self.table_files_query.insert().values(**params)
        self.engine.execute(statement)
        return self

    def get(self):
        find = self.table_files_query.select()
        return self.engine.execute(find).fetchall()

if __name__ == '__main__':

        driver = Driver()
        doc = driver.connect().get()
        doc = [ list(x) for x in doc ]
        doc = pd.DataFrame(doc).to_csv('benchmarks.csv', index=False)
        driver.engine.close()