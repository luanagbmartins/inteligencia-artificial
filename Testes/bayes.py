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

DATASET = 'dataset_reduzido.csv'

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

    while True:

        dataset_gaussian = pd.read_csv('dataset_gaussian.csv')

        X = dataset_gaussian.iloc[:, 0:-1].values.tolist()
        y = np.array(dataset_gaussian.iloc[:, -1].values.tolist()).flatten()


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)

        gaussian_predictions = gnb.predict(X_test)

        dataset_bernoulli = pd.read_csv('dataset_bernoulli.csv')

        X = dataset_bernoulli.iloc[:, 0:-1].values.tolist()
        y = np.array(dataset_bernoulli.iloc[:, -1].values.tolist()).flatten()


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

        bnb = BernoulliNB()
        bnb.fit(X_train, y_train)

        bernoulli_predictions = bnb.predict(X_test)

        col1 = gaussian_predictions.reshape(-1,1)
        col2 = bernoulli_predictions.reshape(-1,1)
        col3 = y_test.reshape(-1,1)
        col1 = pd.DataFrame(col1).astype(int)
        col2 = pd.DataFrame(col2)
        col3 = pd.DataFrame(col3)
        combination = pd.concat([col1, col2, col3], axis=1)

        X = combination.iloc[:, 0:-1].values.tolist()
        y = np.array(combination.iloc[:, -1].values.tolist()).flatten()


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

        combined_bnb = BernoulliNB()
        combined_bnb.fit(X_train, y_train)

        combined_predictions = combined_bnb.predict(X_test)

        cm = confusion_matrix(y_test, combined_predictions)

        params = {
            'id' : str(uuid.uuid4()),
            'name' : 'naive-bayes-02-07-2019',
            '0,0' : float(cm[0][0]),
            '0,1' : float(cm[0][1]),
            '1,0' : float(cm[1][0]),
            '1,1' : float(cm[1][1]),
        }
        
        driver = Driver()
        driver.connect().insert(params)
        driver.engine.close()