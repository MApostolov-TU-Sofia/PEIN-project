import os
from tkinter import *
from tkinter import messagebox
from joblib import load
import numpy as np
import warnings
import json

warnings.filterwarnings("ignore")
BASE_DIR = os.path.join(os.path.dirname(__file__))

def config():
    with open(os.path.join(os.path.abspath(os.path.join(BASE_DIR)), 'config.json')) as f:
        data = json.load(f)
        return data
conf_file = config()

os.chdir(os.path.join(os.path.abspath(os.path.join(BASE_DIR, '..', 'Appl', 'model', 'joblib_models'))))

model = load(conf_file['model'] + '_model.joblib')
transformer = load(conf_file['model'] + '_transformer.joblib')

root = Tk()
root.title('Predict Value')
root.resizable(0, 0)

l_temp = Label(root, text='Temp(C)')
l_temp.grid(row=0)
l_hum = Label(root, text='Hum(%)')
l_hum.grid(row=1)
l_press = Label(root, text='Press(hPa)')
l_press.grid(row=2)
l_no = Label(root, text='NO(ug/m3)')
l_no.grid(row=3)
l_no2 = Label(root, text='NO2(ug/m3)')
l_no2.grid(row=4)
l_ozone = Label(root, text='Ozone(ug/m3)')
l_ozone.grid(row=5)
l_pm10 = Label(root, text='PM10(ug/m3)')
l_pm10.grid(row=6)

l_no_no2_norm = Label(root, text='NO/NO2 norm is:%d' % 200)
l_no_no2_norm.grid(row=3, column=2)
l_o3_norm = Label(root, text='O3 norm is:%d' % 200)
l_o3_norm.grid(row=5, column=2)
l_pm10_norm = Label(root, text='PM10 norm is:%d' % 50)
l_pm10_norm.grid(row=6, column=2)

e_temp = Entry(root)
e_temp.grid(row=0, column=1)
e_hum = Entry(root)
e_hum.grid(row=1, column=1)
e_press = Entry(root)
e_press.grid(row=2, column=1)
e_no = Entry(root)
e_no.grid(row=3, column=1)
e_no2 = Entry(root)
e_no2.grid(row=4, column=1)
e_ozone = Entry(root)
e_ozone.grid(row=5, column=1)
e_pm10 = Entry(root)
e_pm10.grid(row=6, column=1)


def fn_b_predict():
    e_no.delete(0, END)
    e_no2.delete(0, END)
    e_ozone.delete(0, END)
    e_pm10.delete(0, END)
    try:
        arr = np.array([[float(e_temp.get()), float(e_hum.get()), float(e_press.get())]])
        i_tr = transformer.transform(arr)
        i_mdl = model.predict(i_tr)

        e_no.insert(0, round(i_mdl[0][0], 2))
        e_no2.insert(0, round(i_mdl[0][1], 2))
        e_ozone.insert(0, round(i_mdl[0][2], 2))
        e_pm10.insert(0, round(i_mdl[0][3], 2))
    except ValueError:
        messagebox.showinfo('Wrong Value', 'Please enter float values!')


b_predict = Button(root, text='Predict', command=fn_b_predict)
b_predict.grid(row=7)

root.mainloop()