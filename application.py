import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import ridge regressor model and standard scaler pickle files
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_data():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_scaled_data = standard_scaler.transform([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_scaled_data)
        
        return render_template('home.html', results=float(result[0]))
        
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")