import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge
import joblib
import Appl.model as ml

BASE_DIR = os.path.join(os.path.dirname(__file__))


class Model:
    def model_init(self, transformed_data):
        conf_file = ml.App.config(self)
        ml_array = []

        for i, i_file in enumerate(conf_file['i_sheets']):
            tr_d = transformed_data[i]
            regr_model = MultiOutputRegressor(BayesianRidge(max_iter=conf_file['max_iter']), n_jobs=conf_file['n_jobs'])
            regr_model = regr_model.fit(tr_d['X_train'].to_numpy(), tr_d['y_train'])
            joblib.dump(regr_model,
                        os.path.join(os.path.abspath(os.path.join(BASE_DIR, '..', 'joblib_models', i_file + '_model.joblib'))))

            ml_array.append({
                'name': i_file,
                'regr_model': regr_model
            })

        return ml_array
