from io import open_code
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import joblib
import jinja2

app = Flask(__name__)



model = joblib.load(open("desease.pkl",'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/")
def css():
    return render_template('style.css')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    feature_name=['Age', 'Sex', 'cp', 'BP', 'Cholesterol', 'FastingBS', 'ECG', 'MaxHR',
       'exercise', 'Oldpeak', 'ST_Slope']

    df = pd.DataFrame(features_value, columns=feature_name)
    output = model.predict(df)

    if output == 0:
        res_val = "** no heart attack **"
    else:
        res_val = "heart attack"

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


app.run(debug=True)
