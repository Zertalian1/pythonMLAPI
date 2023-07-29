import numpy as np

import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


class PredictProductsCount:
    def __init__(self):
        self.model = self.restore_model()
        self.data = self.__load_data()
        columns = ['Eggs Food', 'Eggs Production']
        self.scaler = self.__data_scaler()
        self.columns = columns

    @staticmethod
    def __data_scaler():
        scaler = MinMaxScaler()
        return scaler.fit(pd.read_csv('./data/prodCount_all.csv', encoding='latin-1').drop(columns=['Country']))

    @staticmethod
    def restore_model():
        return tf.keras.models.load_model('./models/predictProdCount.h5')

    def predict_results(self, year=None, peoples=None):
        t_data = self.data
        if year is not None:
            t_data.at[0, 'Year'] = year
            t_data['Year'] = t_data['Year'].astype(np.int64)
        if peoples is not None:
            t_data.at[0, 'Peoples'] = peoples
            t_data['Peoples'] = t_data['Peoples'].astype(np.int64)
        data = pd.DataFrame(data=self.scaler.transform(t_data))
        pred = self.model.predict(data)
        return pd.DataFrame(pred, columns=self.columns)

    @staticmethod
    def __load_data():
        return pd.read_csv('./data/prodCount_2020.csv', encoding='latin-1')
