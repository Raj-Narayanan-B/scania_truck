import shutil  # type: ignore
from flask import Flask, render_template, request, redirect, url_for  # noqa
from src.constants import SCHEMA_PATH, BUCKET, TEST_DATA, TRAIN_DATA
import pandas as pd
import os
from src.utils import load_yaml, get_files_list_in_s3
from src.pipeline.prediction_pipeline import Prediction_Pipeline
from src.components.stage_0_data_DB_upload import file_tracker_component
from pprint import pprint  # noqa # type:ignore
app = Flask(__name__)

data = {}
columns_dict = load_yaml(SCHEMA_PATH)['Features']
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
    s3_obj = file_tracker_component()
    if request.method == "GET":
        return render_template('Bulk_Prediction.html')
    else:
        if 'file_1' not in request.files:
            return "No file part"
        file = request.files['file_1']
        if file.filename == '':
            return "No selected file"
        try:
            file_lineage = load_yaml(s3_obj.get_data_path_config().data_from_s3)
            if file not in list(file_lineage['files_to_predict'].values()):
                if file in list(file_lineage['files_predicted'].values()):
                    return render_template('Batch_exception.html', result=f"The file: {file} has already been predicted")
                else:
                    s3_obj.file_lineage_tracker(add_list=[file])
                    data_ = pd.read_csv(file)
                    y_pred = y_prediction(data_)
                    s3_obj.file_lineage_tracker(update_list=[file])
                    return render_template('Prediction.html', result=f"{y_pred}")

        except Exception as e:
            return f"Error reading CSV file: {e}"


@app.route("/S3_bucket_prediction", methods=['GET', 'POST'])
def s3_bucket():
    s3_obj = file_tracker_component()
    if request.method == "GET":
        s3_files_dict = s3_obj.s3.list_objects_v2(Bucket=BUCKET)['Contents']
        s3_files_list = get_files_list_in_s3(s3_files_dict)
        s3_files_list.remove(TEST_DATA)
        s3_files_list.remove(TRAIN_DATA)
        print(s3_files_list)
        return render_template('S3_bucket.html', s3_file_list=s3_files_list)
    else:
        try:
            selected_file = request.form.get('selected_file')
            file_lineage = load_yaml(s3_obj.get_data_path_config().data_from_s3)

            if selected_file in list(file_lineage['files_to_predict'].values()):
                filepath = os.path.join(f"{s3_obj.get_data_path_config().temp_dir_root}", f"{selected_file}")

                s3_obj.s3_data_download(key=selected_file,
                                        filepath=filepath)

                data_ = pd.read_csv(filepath)
                y_pred = y_prediction(data_)

                s3_obj.file_lineage_tracker(update_list=[selected_file])

                shutil.rmtree(s3_obj.get_data_path_config().temp_dir_root)

                return render_template('Prediction.html', result=f"{y_pred}")
            elif selected_file in list(file_lineage['files_predicted'].values()):
                return render_template('S3_exception.html', result=f"The file: {selected_file} has already been predicted")
        except Exception as e:
            return f"Error: {e}"


@app.route("/Airflow_server", methods=['GET'])
def airflow():
    if request.method == 'GET':
        airflow_server_link = "http://localhost:8080"
        return redirect(airflow_server_link)


def y_prediction(data_):
    prediction_obj = Prediction_Pipeline(data=data_)
    y_pred = prediction_obj.prediction_pipeline()
    return y_pred


if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=1234)
