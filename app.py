from flask import Flask, render_template, request, redirect # noqa
from src.constants import SCHEMA_PATH
from src.utils import load_yaml
from src.pipeline.prediction_pipeline import Prediction_Pipeline

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict_form", methods=['GET', 'POST'])
def prediction():
    data_ = {}
    columns_dict = load_yaml(SCHEMA_PATH).Features
    columns = list(columns_dict.keys())
    for i in columns:
        data_[i] = request.form[i]
    prediction_obj = Prediction_Pipeline(data=data_)
    y_pred = prediction_obj.prediction_pipeline()
    return render_template('result.html', result=y_pred)


if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=1234)
