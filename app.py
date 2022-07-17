import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from flask import Flask, render_template, request

model = tf.keras.models.load_model('dnn.h5')

# Load minMaxScaler
with open('minMaxScaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')


@app.route('/home', methods=['POST'])
def home():
    cr = np.float32(request.form['cr'])
    rlz = np.float32(request.form['rlz'])
    nba = np.float32(request.form['nba'])
    crtb = np.float32(request.form['crtb'])
    noc = np.float32(request.form['noc'])
    nor = np.float32(request.form['nor'])
    age = np.float32(request.form['age'])
    dte = np.float32(request.form['dte'])
    ha = np.float32(request.form['ha'])
    proptr = np.float32(request.form['proptr'])
    ptr = np.float32(request.form['ptr'])
    li = np.float32(request.form['li'])
    arr = np.array([[cr, rlz, nba, crtb, noc, nor, age, dte, ha, proptr, ptr, li]])
    print(arr)
    arr = scaler.transform(arr)
    pred = model.predict(arr)
    print(pred)
    return render_template('home.html', data=pred[0][0])

if __name__ == "__main__":
    app.run(debug=True)