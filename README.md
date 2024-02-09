# scania_truck

Things to include finally -
    * Hyper-Param training using Optuna and HyperOpt
    * Use DVC to monitor data stored in gdrive
    * Use MLFLOW to track model performance
    * use Pytest to create unit tests
    * Use tox to automate testing
    * Use Docker to finally encapsulate the entire process
    * Use Airflow to schedule pipeline workflows
    * Use GitHub Actions (.github/workflows) to run in github

Basic_setup:
- Check Dataset size
- 
- Astra DataBase Variables
- 
- Data Validation
    * Check d_types
    * Check column names
    
To enable auto formatting on save a document. create a .vscode directory if not already present.
Then, create a settings.json file.
And in it, type the following:
// .vscode/settings.json
{
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--max-line-length=200"
    ],
    "python.formatting.provider": "autopep8",
    "editor.formatOnSave": true,
    "python.formatting.autopep8Args": [
        "--max-line-length=200"
    ],
    "python.analysis.typeCheckingMode": "off"
}
==> This is only for the current project. If you want to enable this for the entire VSCode globally, then goto settings from file>preferences. And in the search bar, type: JSON.

You will get an option with the hyperlink to json settings. This will be the default settings.json file for VSCode. In it, add the same above json format (without the {} and save the file). Auto-Formatting as per autopep8 will be enabled.

Logistic_Regression:
SGD_Classifier:
Random Forest:
Ada_Boost:
Grad_Boost:
Bagging_Classifier:
ExtraTreesClassifier:
Hist_Grad_Boost_Classifier:
Decision_Tree_Classifier:
XGB_Classifier:
KNN_Classifier:
Stacked_Classifier:

Artifacts so far:
    - Raw Data 6x (3 train and 3 test)
    - Basic Preprocessed data (train, test) (also the merged data)
    - Final processed training data (split into train and validation) -> this is gotten from the training set of basic preprocessed data
    - The preprocessor.joblib
    - The Champion Model


Artifacts to saved from each component:
    - Stage1 - Data_ingestion
        - Should save: Raw data (3x Train Data, 3x Test Data)

    - Stage2 - Initial Preprocessing
        - Should save both the preprocessed data (train, test)

    - Stage4 -  Final Preprocessing
        - Should save the processed/transformed data from train_set of Stage2-Initial Preprocessing module
          This will have the training and validation data

    - Stage5 - Model Tuning, Tracking, Training
        - Should save the preprocessor.joblib

    - Stage6 - Test Data Prediction
        - Should save the champion model

export MLFLOW_TRACKING_URI=https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow \
export MLFLOW_TRACKING_USERNAME=Raj-Narayanan-B \
export MLFLOW_TRACKING_PASSWORD=8af4cc66be8aec751397fd525e47ae395fa67442

export MLFLOW_TRACKING_URI=https://dagshub.com/Raj-Narayanan-B/scania_truck.mlflow \
export MLFLOW_TRACKING_USERNAME=Raj-Narayanan-B \
export MLFLOW_TRACKING_PASSWORD=8af4cc66be8aec751397fd525e47ae395fa67442



TO-DO:
Feb8,2024:
    <!-- - update outs section in dvc.yaml file for the stage: model_tuning_tracking_training
    - the placeholders should be updated.
    - the trials df from parameter_tuning2 should be given as an outs
    - rest all that are being saved from model_tuning_tracking_training.py file should be given in outs -->
    <!-- - update the way mlflow_model_sources.yaml file is being created from model_tuning_tracking_training.py file. The keys should be changed accordingly. -->

    <!-- - udpdate the outs and deps in test_data_prediction stage according to the placeholders.
    - update dyc.yaml file's CMD sections -->
    - resume the databases & try dvc repro -f dvc.yaml


    <!-- - if you get an error to check the cluster status, delete the vector databases and create new vectorless databases and upload the data into them using dsbulkloader.

    - save the token.json and secure_connect_bundles of those new databases
    - try dvc repro -f dvc.yaml again! -->

Feb 9, 2024
    - create prediction pipeline
    - check if any line of code has the full file path. It should be only the relative path.
    - create app.py and its dependencies (static & templates)
    - create docker image