import os
import numpy as np
import json
from Appl.model.data.read_file import ReadFile as rf
from Appl.model.data.merge_file import MergeFile as mf
from Appl.model.transformer.RobustScalerTransform import Transformer as tr
from Appl.model.models_init.BayesianRidgeModel import Model as brm
from Appl.model.test.ModelTest import Test as tst

BASE_DIR = os.path.join(os.path.dirname(__file__))


class App:
    def config(self):
        with open(os.path.join(os.path.abspath(os.path.join(BASE_DIR, '..')), 'config.json')) as f:
            data = json.load(f)
            return data

    def check_file_exist(self, conf_file):
        i_files_exist = True
        for i_file in conf_file['i_sheets']:
            if (not os.path.isfile(os.path.join(os.path.abspath(os.path.join(BASE_DIR, '..', 'documents', i_file + '.csv'))))):
                i_files_exist = False
                break
        return i_files_exist

    def run(self):
        conf_file = self.config()
        i_files_exist = self.check_file_exist(conf_file)
        f_dateframe_arrays = None

        if (not i_files_exist):
            f_dateframe_arrays = rf.read(self,
                    os.path.abspath(os.path.join(BASE_DIR, '..', 'data')),
                    conf_file['i_files'],
                    conf_file['i_sheets'],
                    conf_file['i_usecols'],
                    [{'O3': np.single, 'PM10': np.single}, None])
            f_dateframe_arrays = mf.merge(self,
                                      f_dateframe_arrays,
                                      conf_file['i_sheets'])

        transform_data = tr.transform_data(self)
        model_data = brm.model_init(self, transform_data)
        tst.model_test(self, transform_data, model_data)

        print('Done!')