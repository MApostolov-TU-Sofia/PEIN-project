import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__))


class MergeFile:
    def merge(self, df_array, i_sheets):
        dtDates = df_array[0]
        dtHours = df_array[1]
        df_merged = []
        for j, frs in enumerate(i_sheets):
            df = pd.DataFrame()
            for i in range(dtDates[frs]['Date'].size):
                iDate = dtDates[frs]['Date'][i] if ('Date' in dtDates[frs].columns) else None
                iO3 = dtDates[frs]['O3'][i] if ('O3' in dtDates[frs].columns) else 0
                iPM10 = dtDates[frs]['RM10'][i] if ('RM10' in dtDates[frs].columns) else 0

                iHoursFromDay = dtHours[frs][dtHours[frs]['Date'].str.contains(iDate)]
                if (not 'Press' in iHoursFromDay.columns):
                    iHoursFromDay = iHoursFromDay.assign(Press=0)
                iHoursFromDay.insert(6, 'O3', iO3)
                iHoursFromDay.insert(7, 'PM10', iPM10)
                df = df.append(iHoursFromDay, ignore_index=False)

            df = df.dropna()
            df = df.round({'N0': 2, 'NO2': 2, 'O3': 2, 'PM10': 2, 'AirTemp': 1, 'UMR': 1, 'Press': 0})

            df.to_csv(os.path.join(os.path.abspath(os.path.join(BASE_DIR, '../..')), 'documents', frs + '.csv'),
                      columns=['NO', 'NO2', 'AirTemp', 'Press', 'UMR', 'O3', 'PM10'])
            df_merged.append(df)

        return df_merged