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
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from sklearn.metrics import confusion_matrix

DATASET = 'dataset.csv'

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

import datetime

if __name__ == '__main__':

    dataset = pd.read_csv(DATASET)
    driver = Driver()
    driver.connect()
    start = int(datetime.datetime.now().timestamp())
    while True:

        now = int(datetime.datetime.now().timestamp())
        if now - start > 60 * 5:
            break

        X = dataset.iloc[:, 0:-1].values
        y = dataset.iloc[:, -1].values

        labels = LabelEncoder()
        y = labels.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Initialising the ANN
        classifier = Sequential()

        # Adding the Single Perceptron or Shallow network
        classifier.add(Dense(output_dim=64, 
                            init='uniform', 
                            activation='relu', 
                            input_dim=18))

        # Adding dropout to prevent overfitting
        classifier.add(Dropout(p=0.1))

        # Adding the output layer
        classifier.add(Dense(output_dim=1, 
                            init='uniform', 
                            activation='sigmoid'))

        # criterion loss and optimizer 
        classifier.compile(optimizer='rmsprop',
                        loss='binary_crossentropy', 
                        metrics=['accuracy'])

        # Fitting the ANN to the Training set
        classifier.fit(X_train, 
                    y_train, 
                    batch_size=100, 
                    nb_epoch=80)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)


        params = {
            'id' : float(datetime.datetime.now().timestamp()),
            'name' : 'new_ann-02-07-2019',
            '0,0' : float(cm[0][0]),
            '0,1' : float(cm[0][1]),
            '1,0' : float(cm[1][0]),
            '1,1' : float(cm[1][1]),
        }
        
        driver.insert(params)