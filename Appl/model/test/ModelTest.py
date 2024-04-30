import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge
import joblib
import matplotlib
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__))


class Test:
    def model_test(self, transform_data, model_data):
        for i, mdl in enumerate(model_data):
            score = mdl['regr_model'].score(transform_data[i]['X_test_transformed'], transform_data[i]['y_test'])
            print(score)

            pred = mdl['regr_model'].predict(transform_data[i]['X_test'])
            print(pred)

            default_x_ticks = range(pred.shape[0])
            plt.plot(default_x_ticks, pred, lw=0.8, color='blue', label=mdl['name'] + ' predicted regression')
            plt.legend()
            plt.savefig(os.path.join(os.path.abspath(os.path.join(BASE_DIR, '..', 'test_images', mdl['name'] + '_predicted_regression.png'))))
            plt.show()
