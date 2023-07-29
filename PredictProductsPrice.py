
import pandas as pd
import tensorflow as tf


class PredictProductsPrice:
    def __init__(self):
        self.model = self.restore_model()
        self.data = self.__load_data()
        columns = []
        for name in self.data.drop(columns=['Year']).columns:
            columns.append(name)
        self.columns = columns

    @staticmethod
    def restore_model():
        return tf.keras.models.load_model('./models/minProd.h5')

    def predict_results(self, year):
        self.data.at[0, 'Year'] = year
        t_data = self.data.drop(columns=['Year'], axis=1)
        t_data.index = self.data['Year']
        pred = self.model.predict(t_data.to_numpy())
        return pd.DataFrame(pred, columns=self.columns)

    @staticmethod
    def __load_data():
        return pd.read_csv('./data/minBag_2020.csv', encoding='latin-1')
