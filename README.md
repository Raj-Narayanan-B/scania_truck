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

ENV:
- mlflow
- dvc
- astradb
- airflow
- aws secrets
- azure secrets

TO-DO:
Feb8,2024:
    - update outs section in dvc.yaml file for the stage: model_tuning_tracking_training
    - the placeholders should be updated.
    - the trials df from parameter_tuning2 should be given as an outs
    - rest all that are being saved from model_tuning_tracking_training.py file should be given in outs -->
    - update the way mlflow_model_sources.yaml file is being created from model_tuning_tracking_training.py file. The keys should be changed accordingly.

    - udpdate the outs and deps in test_data_prediction stage according to the placeholders.
    - update dyc.yaml file's CMD sections
    - resume the databases & try dvc repro -f dvc.yaml


    - if you get an error to check the cluster status, delete the vector databases and create new vectorless databases and upload the data into them using dsbulkloader.

    - save the token.json and secure_connect_bundles of those new databases
    - try dvc repro -f dvc.yaml again!

Feb 9, 2024
    - create prediction pipeline
    - check if any line of code has the full file path. It should be only the relative path.
    - create app.py and its dependencies (static & templates)

Feb 13, 2024
    - clean up the templates(compulsarily) and static files(compulsarily) and app.pyy file(if necessary)
    - Setup data upload into Astra DB
    - Configure the AirFlow Server using docker
    - clean the training_pipeline.py file (put everything inside a single function as before) #NOT NEEDED
    - check if the env variables are set in the docker file


Feb 14, 2024
    - Create the github/workflows/main.yaml
    - make sure that dvc is tracking all the artifact files (all the csv files created and downloaded, joblib files(model/preprocessor), params, config, schema, AstraDB secrets files)
    - test the entire app


CREATE DATABASE airflow_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'raj' IDENTIFIED BY 'admin';
GRANT ALL PRIVILEGES ON airflow_db.* TO 'raj';

Docker Airflow Container Run Command:
docker run -p 8080:8080 -v F:/iNeuron/Projects/scania_failures_2/airflow/dags:/app/airflow/dags scania_truck:latest


Data_Validation
in 0_trial.ipynb
- Duplicates check 
- Check for columns with 0 std_dev()
- Drop columns that have more than 50% of missing values
- check for histogram features
- check for PCA # not needed
- Create the final schema and ensure that the user inputting the new values have the columns present in final schema

*** Change the module: stage4_final_preprocessing to data validation or 
    create a new function inside that module named data_validation that includes all 
    steps mentioned above.

- Change the entity config and it's dataclass names to "artifact"
- Make changes in training pipeline (include data validation)
- Make changes in prediction pipeline (include data validation)


- add files_tracker at the end of prediction pipeline
- add s3 bucket option in index.html
- if any file is being predicted using the S3 option in webpage, a temp folder should be created and after the prediction, it should be removed.
- if any file is manually selected by the user, it should be added in the files_tracker