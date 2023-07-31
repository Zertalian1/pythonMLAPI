from functools import cmp_to_key

import numpy as np

import pandas as pd
import tensorflow as tf

from os import listdir
from os.path import isfile, join
import locale

from sklearn.preprocessing import MinMaxScaler


class PredictProductsCount:
    def __init__(self):
        self.model = self.__restore_models()
        self.all_data = self.__load_all_data()
        self.scalers = self.__restore_scalers()
        self.allProdNames = self.__load_all_names()

    @staticmethod
    def __load_all_names():
        products = ['Alcoholic Beverages', 'Barley and products', 'Bovine Meat', 'Cereals - Excluding Beer', 'Eggs',
                    'Fish, Seafood', 'Fruits, other', 'Meat', 'Milk - Excluding Butter', 'Nuts and products',
                    'Oil and animal fat', 'Peas', 'Pigmeat', 'Potatoes and products', 'Poultry Meat',
                    'Rice and products', 'Rye and products', 'Sugar & Sweeteners', 'Tea (including mate)', 'Vegetables',
                    'Wheat and products']
        fields = []
        for product in products:
            fields.append("{} Food".format(product))
            fields.append("{} Production".format(product))
        return fields

    @staticmethod
    def __restore_models():
        files_names = [f for f in listdir('./models/predProdCount/') if isfile(join('./models/predProdCount/', f))]
        files_names = sorted(files_names, key=cmp_to_key(locale.strcoll))
        models = []
        for file_name in files_names:
            models.append(tf.keras.models.load_model('models/predProdCount/{}'.format(file_name)))
        return models

    @staticmethod
    def __restore_scalers():
        files_names = [f for f in listdir('./data/prodCountAllData/') if isfile(join('./data/prodCountAllData/', f))]
        files_names = sorted(files_names, key=cmp_to_key(locale.strcoll))
        scalers = []
        for file_name in files_names:
            scaler = MinMaxScaler()
            scalers.append(
                scaler.fit(
                    pd.read_csv(
                        'data/prodCountAllData/{}'.format(file_name), encoding='latin-1'
                    ).drop(columns=['Country'])
                )
            )
        return scalers

    @staticmethod
    def __load_all_data():
        files_names = [f for f in listdir('./data/prodCount2020/') if isfile(join('./data/prodCount2020/', f))]
        files_names = sorted(files_names, key=cmp_to_key(locale.strcoll))
        all_data = []
        for file_name in files_names:
            all_data.append(pd.read_csv('./data/prodCount2020/{}'.format(file_name), encoding='latin-1'))
        return all_data

    def predict_results_all_prod(self, year=None, peoples=None):
        columns = []
        for prod in self.allProdNames:
            if prod == 'Bovine Meat Production':
                continue
            columns.append(prod)
        pred = [[]]
        for i in range(0, len(self.all_data)):
            t_data = self.all_data[i]
            if year is not None:
                t_data.at[0, 'Year'] = year
                t_data['Year'] = t_data['Year'].astype(np.int64)
            if peoples is not None:
                t_data.at[0, 'Peoples'] = peoples
                t_data['Peoples'] = t_data['Peoples'].astype(np.int64)
            data = pd.DataFrame(data=self.scalers[i].transform(t_data))
            prediction = self.model[i].predict(data)
            if i == 2:
                pred[0].append(prediction[0][0])
                continue
            pred[0].append(prediction[0][0])
            pred[0].append(prediction[0][1])
        return pd.DataFrame(pred, columns=columns)

    def predict_results_selected_prod(self, products, year=None, peoples=None):
        columns = []
        for i in products:
            if i == 2:
                columns.append(self.allProdNames[2 * i])
                continue
            columns.append(self.allProdNames[2 * i])
            columns.append(self.allProdNames[2 * i + 1])
        pred = [[]]
        for i in products:
            t_data = self.all_data[i]
            if year is not None:
                t_data.at[0, 'Year'] = year
                t_data['Year'] = t_data['Year'].astype(np.int64)
            if peoples is not None:
                t_data.at[0, 'Peoples'] = peoples
                t_data['Peoples'] = t_data['Peoples'].astype(np.int64)
            data = pd.DataFrame(data=self.scalers[i].transform(t_data))
            prediction = self.model[i].predict(data)
            if i == 2:
                pred[0].append(prediction[0][0])
                continue
            pred[0].append(prediction[0][0])
            pred[0].append(prediction[0][1])
        return pd.DataFrame(pred, columns=columns)
