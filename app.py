from flask import Flask, render_template, request, redirect, url_for  # noqa
from src.constants import SCHEMA_PATH
import pandas as pd
from src.utils import load_yaml
from src.pipeline.prediction_pipeline import Prediction_Pipeline
from pprint import pprint  # noqa # type:ignore
app = Flask(__name__)

data = {}
columns_dict = load_yaml(SCHEMA_PATH).Features
a_series__columns = [s for s in columns_dict if s.startswith('a')]
b_series__columns = [s for s in columns_dict if s.startswith('b')]
c_series__columns = [s for s in columns_dict if s.startswith('c')]
d_series__columns = [s for s in columns_dict if s.startswith('d')]
e_series__columns = [s for s in columns_dict if s.startswith('e')]


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/home")
def home_page():
    return render_template('index.html')


@app.route("/A_series", methods=['GET', 'POST'])
def a_series():
    if request.method == "GET":
        return render_template('A_series.html')
    else:
        for i in a_series__columns:
            data[i] = request.form[i]
        # pprint(data, compact=True)
        return redirect(url_for('b_series'))


@app.route("/B_series", methods=['GET', 'POST'])
def b_series():
    if request.method == "GET":
        return render_template('B_series.html')
    else:
        for i in b_series__columns:
            data[i] = request.form[i]
        # pprint(data, compact=True)
        return redirect(url_for('c_series'))


@app.route("/C_series", methods=['GET', 'POST'])
def c_series():
    if request.method == "GET":
        return render_template('C_series.html')
    else:
        for i in c_series__columns:
            data[i] = request.form[i]
        # pprint(data, compact=True)
        return redirect(url_for('d_series'))


@app.route("/D_series", methods=['GET', 'POST'])
def d_series():
    if request.method == "GET":
        return render_template('D_series.html')
    else:
        for i in d_series__columns:
            data[i] = request.form[i]
        # pprint(data, compact=True)
        return redirect(url_for('e_series'))


@app.route("/E_series", methods=['GET', 'POST'])
def e_series():
    if request.method == "GET":
        return render_template('E_series.html')
    else:
        for i in e_series__columns:
            data[i] = request.form[i]
        # pprint(data, compact=True)

        y_pred = y_prediction(data)
        return render_template('Prediction.html', result=f"{y_pred}")


@app.route("/Prediction", methods=['GET'])
def result():
    if request.method == "GET":
        return render_template('Prediction.html')


@app.route("/Batch_prediction", methods=['GET', 'POST'])
def predict():
    if request.method == "GET":
        return render_template('Bulk_Prediction.html')
    else:
        if 'file_1' not in request.files:
            return "No file part"
        file = request.files['file_1']
        if file.filename == '':
            return "No selected file"
        try:
            data_ = pd.read_csv(file)
            y_pred = y_prediction(data_)
            return render_template('Prediction.html', result=f"{y_pred}")
        except Exception as e:
            return f"Error reading CSV file: {e}"


def y_prediction(data_):
    prediction_obj = Prediction_Pipeline(data=data_)
    y_pred = prediction_obj.prediction_pipeline()
    return y_pred


if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=1234)
