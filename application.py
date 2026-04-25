from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import pickle files for scaling and model

model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method=="POST":
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        wind_speed = float(request.form.get('wind_speed'))
        rainfall = float(request.form.get('rainfall'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        isi = float(request.form.get('isi'))
        classes = float(request.form.get('classes'))

        new_scaled_data = scaler.transform([[temperature, ffmc, dmc, isi, classes, humidity, wind_speed, rainfall]])

        result = model.predict(new_scaled_data)

        return render_template('home.html', results=round(result[0], 2 ))
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)