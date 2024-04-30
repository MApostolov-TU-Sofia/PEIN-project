import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import Appl.model as ml

BASE_DIR = os.path.join(os.path.dirname(__file__))


class Transformer:
    def transform_data(self):
        conf_file = ml.App.config(self)
        tr_array = []
        for i_file in conf_file['i_sheets']:
            df = pd.read_csv(os.path.join(os.path.abspath(os.path.join(BASE_DIR, '../..', 'documents', i_file + '.csv'))))

            X = df[['AirTemp', 'Press', 'UMR']]
            y = df[['NO', 'NO2', 'O3', 'PM10']]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=conf_file['test_size'], train_size=conf_file['train_size'], random_state=conf_file['random_state'])

            transformer = RobustScaler().fit(X_train)
            X_train_transformed = transformer.transform(X_train)
            X_test_transformed = transformer.transform(X_test)

            joblib.dump(transformer,
                        os.path.join(os.path.abspath(os.path.join(BASE_DIR, '..', 'joblib_models', i_file + '_transformer.joblib'))))
            tr_array.append({
                'X': X,
                'y': y,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'transformer': transformer,
                'X_train_transformed': X_train_transformed,
                'X_test_transformed': X_test_transformed
            })

        return tr_array
