from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction import CustomData, PredictionPipeline


app = Flask(__name__)


@app.route('/', methods = ['GET', "POST"])
def home():
    return render_template("index.html")


@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            age = int(request.form.get('age')),
            cp = request.form.get('cp'),
            trtbps = request.form.get('trtbps'),
            oldpeak= float(request.form.get('oldpeak')),
            chol = int(request.form.get('chol,')),
            fbs = request.form.get('fbs'),
            restecg= request.form.get('restecg'),
            thall = request.form.get('thall'),
            slp = int(request.form.get('slp')),
            thalachh= int(request.form.get('thalachh')),
            exng = request.form.get('exng'),
            caa = request.form.get('caa')
        )

        pred_df = data.get_data_as_data_frame()
        prediction_pipeline = PredictionPipeline()
        results = prediction_pipeline.predict(pred_df)
        result_string = f"Predicted chances of heart attack: {round(results[0], 2) * 100}%"
        return render_template('index.html',results= result_string)

if __name__ == "__main__":
    app.run(debug= True)