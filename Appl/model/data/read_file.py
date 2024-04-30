import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__))


class ReadFile:
    def read(self, i_path, i_files, i_sheets, i_usecols, i_dtype):
        os.chdir(i_path)
        df = []
        for i, file in enumerate(i_files, start=0):
            df.append(pd.read_excel(file,
                           sheet_name=i_sheets,
                           header=1,
                           usecols=i_usecols[i],
                           dtype=i_dtype[i]))

        return df
