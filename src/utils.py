import pandas as pd
import numpy as np
from pathlib import Path #type: ignore
from box import ConfigBox
from ensure import ensure_annotations
import os
import joblib
import yaml
from src import logger
from src.constants import *
import json
from typing import NewType #type: ignore
ML_Model = NewType('Machine_Learning_Model', object)

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import warnings as w #type: ignore
w.filterwarnings('ignore')

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split
# from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             accuracy_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, 
                              HistGradientBoostingClassifier, StackingClassifier, VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV
import optuna
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
# import ruamel.yaml as yaml

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
from mlflow.client import MlflowClient
client = MlflowClient(tracking_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow",
                      registry_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow")
trial_number = 0

def load_yaml(filepath:Path):
    try:
        filepath_,filename = os.path.split(filepath)
        with open(filepath) as yaml_file:
            config = yaml.load(yaml_file,
                               Loader = yaml.CLoader)
            logger.info(f"{filename} yaml_file is loaded")
            return ConfigBox(config)
    except Exception as e:
        raise e
    
def save_yaml(file,filepath:Path):
    try:
        yaml.dump(data = file,
                  stream = open(filepath,'w'),
                  indent = 4)
        logger.info("yaml file is saved")
    except Exception as e:
        raise e
    
def load_binary(filepath:Path):
    try:
        object = joblib.load(filename=filepath)
        logger.info(f"pickled_object: {filepath} loaded")
        return object
    except Exception as e:
        raise e

def save_binary(file,filepath:Path):
    try:
        joblib.dump(file,filepath)
        logger.info(f"object: {filepath} pickled")
    except Exception as e:
        raise e
    
def mk_dir(filepath:Path):
    os.makedirs(filepath, exist_ok=True)
    logger.info(f"directory: {filepath} created")

def DB_data_loader(config: list):
    with open(config[1]) as f:
        secrets = json.load(f)
        logger.info(f"{config[1]} json file is loaded")
    
    cloud_config = {list(config[0].keys())[0] : list(config[0].values())[0]}
    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]
    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)

    session = cluster.connect()

    logger.info(f"connected to cluster/keyspace: {config[2]}")

    keyspace = config[2]
    table = config[3]
    query = f"SELECT * FROM {keyspace}.{table}"

    result = session.execute(query)
    logger.info(f"Data queried from keyspace: {config[2]}")
    df = pd.DataFrame(list(result))
    df.to_csv(config[4],index = False)
    print(df.shape)

def stage_1_processing_function(dataframes: list) -> pd.DataFrame:
    logger.info(f"Stage 1 Processing Commencing")

    data_merger_1 = pd.merge(left = dataframes[0],
                              right = dataframes[1],
                              how = 'outer',
                              on = 'ident_id')
    logger.info(f"Data Merging commencing")

    data_merger_final = pd.merge(left = data_merger_1,
                                  right = dataframes[2],
                                  how = 'outer',
                                  on = 'ident_id')
    logger.info(f"Data Merging complete")

    data_merger_final = data_merger_final.sort_values(by='ident_id').reset_index(drop=True)
    logger.info(f"Sorting and reseting_index complete")

    data_merger_final.drop(columns='ident_id',inplace=True)
    logger.info(f"Dropping column: 'ident_id'")

    data_merger_final.rename(columns={'field_74_':'class'},inplace=True)
    logger.info(f"Renaming Target Column")

    data_merger_final['class'] = data_merger_final['class'].map({'neg':0,'pos':1})
    logger.info(f"Mapping Target Column values")

    data_merger_final.replace('na',np.nan,inplace=True)
    logger.info(f"Replacing 'na' to 'np.nan' values")

    test_col_list = [i for i in data_merger_final.columns if i != 'class']
    logger.info(f"Creating list of column names of input features")

    for i in test_col_list:
        data_merger_final[i]=data_merger_final[i].astype('float')
    logger.info(f"dtype of input features converted from 'object' to 'float'")

    logger.info(f"Stage 1 processing complete - Returning processed dataframe")
    return (data_merger_final)

def schema_saver(dataframe: pd.DataFrame):
    from src.config.configuration_manager import ConfigurationManager
    obj = ConfigurationManager()
    dict_cols={}
    dict_cols['Features'] = {}
    dict_cols['Target'] = {}
    for i in dataframe.columns:
        if i =='class':
            dict_cols['Target'][i] = str(dataframe[i].dtypes)
        else:
            dict_cols['Features'][i]=str(dataframe[i].dtypes)

    save_yaml(file = dict_cols, filepath = obj.schema)

def train_test_splitter(dataframe: pd.DataFrame) -> pd.DataFrame:
    train,test = train_test_split(dataframe,test_size = 0.25,random_state=8)
    return (train,test)

def stage_2_processing_function(dataframe: pd.DataFrame) -> pd.DataFrame:
    from src.config.configuration_manager import ConfigurationManager
    obj = ConfigurationManager()
    preprocessor_config = obj.get_preprocessor_config()
    schema = load_yaml(obj.schema)
    target = list(schema.Target.keys())[0]
    if not os.path.exists(preprocessor_config.preprocessor_path):
        logger.info(f"Stage 2 Processing Commencing")

        pipeline = Pipeline(steps=[('Knn_imputer',KNNImputer()),
                                ('Robust_Scaler',RobustScaler())],
                                verbose=True)
        smote = SMOTETomek(n_jobs=-1,sampling_strategy='minority',random_state=8)

        logger.info(f"Pipeline created with KnnImputer, RobustScaler")
        logger.info(f"SmoteTomek obj created")


        X = dataframe.drop(columns=target)
        # logger.info(f"Creating X dataframe with only the input features - dropping target")

        y = dataframe[target]
        # logger.info(f"Creating y - target")

        logger.info(f"Commencing pipeline transformation")
        X_transformed = pipeline.fit_transform(X = X, y = y)
        logger.info(f"Pipeline transformation complete")

        logger.info(f"Commencing SmoteTomek")
        X_smote,y_smote = smote.fit_resample(X = X_transformed,
                                            y = y)
        logger.info(f"SmoteTomek Complete")
        
        columns_list = list(pipeline.get_feature_names_out())
        X_column_names = [i for i in columns_list if i!=target]
        y_column_name = target

        logger.info(f"Returning the transformed dataframe")
        transformed_df = pd.DataFrame(X_smote,columns = X_column_names)
        transformed_df[y_column_name] = y_smote

        logger.info(f"Saving the pipeline object")
        save_binary(file = pipeline,
                    filepath = preprocessor_config.preprocessor_path)
        logger.info(f"Pipeline saved at: {preprocessor_config.preprocessor_path}")
        
        logger.info(f"Stage 2 Processing Complete")

        return (transformed_df)

    else:
        logger.info(f"Stage 2 Processing Commencing")

        loaded_pipeline = load_binary(preprocessor_config.preprocessor_path)
        smote = SMOTETomek(n_jobs=-1,sampling_strategy='minority',random_state=8)

        logger.info(f"Pipeline loaded & SmoteTomek created")

        X = dataframe.drop(columns = target)
        y = dataframe[target]

        logger.info(f"Commencing pipeline transformation")
        X_transformed = loaded_pipeline.transform(X = X)
        logger.info(f"Pipeline transformation complete")

        logger.info(f"Commencing SmoteTomek")
        X_smote,y_smote = smote.fit_resample(X = X_transformed,
                                             y = y)
        logger.info(f"SmoteTomek Complete")
        
        columns_list = list(loaded_pipeline.get_feature_names_out())
        X_column_names = [i for i in columns_list if i!=target]
        y_column_name = target
        
        logger.info(f"Returning the transformed dataframe")
        transformed_df = pd.DataFrame(X_smote,columns = X_column_names)
        transformed_df[y_column_name] = y_smote

        logger.info(f"Stage 2 Processing Complete")

        return (transformed_df)
    
def eval_metrics(y_true , y_pred) -> float:
        balanced_accuracy_score_ = float(balanced_accuracy_score(y_true=y_true, y_pred=y_pred))

        f1_score_ = float(f1_score(y_true=y_true, y_pred=y_pred))

        accuracy_score_ = float(accuracy_score(y_true=y_true, y_pred=y_pred))

        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
        cost_ = float((10*fp)+(500*fn))
        
        return ({"Balanced_Accuracy_Score" : balanced_accuracy_score_,
                 "F1_Score" : f1_score_,
                 "Accuracy_Score" : accuracy_score_,
                 "Cost" : cost_})

def parameter_tuning(model_class : ML_Model, 
                     model_name: str, 
                     x_train: pd.DataFrame, 
                     x_test: pd.DataFrame, 
                     y_train: pd.DataFrame, 
                     y_test: pd.DataFrame,
                     report_: dict,
                     *args):
       
    tuner_report = {}
    tuner_report['Optuna'] = {}
    tuner_report['HyperOpt'] = {}
    params_config = load_yaml(PARAMS_PATH)
    exp_id_list = []

    tags = {"tuner_1": "optuna",
            "tuner_2": "hyperopt",
            "metrics": "['Balanced_Accuracy_Score', 'F1_Score', 'Accuracy_Score', 'Cost']"} 
    exp_id = client.create_experiment(name = f"61_{model_name}_61", tags = tags) 

####################################################### OPTUNA #######################################################
    with mlflow.start_run(experiment_id = exp_id,
                          run_name = f"Optuna for {model_name}",
                          tags = {"tuner" : "optuna",
                                  "run_type": "parent"}) as optuna_parent_run:
        parent_run_id = optuna_parent_run.info.run_id

        def optuna_objective(trial):
            with mlflow.start_run(experiment_id = exp_id,
                                  run_name = f"Trial {(trial.number)+1} for {model_name} (optuna)",
                                  tags = {"run_type": "child"},
                                  nested = True) as child_run:
                space_optuna = {}
                for key,value in params_config['optuna'][model_name].items():
                    space_optuna[key] = eval(value)
                if model_name == 'Stacked_Classifier':
                    model = model_class.set_params(**space_optuna)
                else:
                    model = model_class(**space_optuna)
                # model.set_params(**space_optuna)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                cost = eval_metrics(y_test , y_pred)["Cost"]
                
                data = (x_train, y_test, y_pred)
                mlflow_logger(data = data,
                              model = model,
                              model_name = model_name,
                            #   params = model.get_params(),
                              should_log_parent_model = False,
                              artifact_path = f'optuna_{model_name}' if model_name == 'XGB_Classifier' else f'optuna_{model_name}')           
                print("Artifacts URI of Optuna Child Run: ",mlflow.get_artifact_uri())
                return cost
            
        print("Artifacts URI of Optuna Parent Run: ",mlflow.get_artifact_uri())
        find_param=optuna.create_study(direction = "minimize")
        find_param.optimize(optuna_objective,n_trials=2)

        data = (x_train, x_test, y_train, y_test)
        mlflow_logger(data = data,
                      model_name = model_name,
                      should_log_parent_model = True,
                      run_id = parent_run_id,
                      exp_id = exp_id,
                    #   registered_model_name = f"Challenger_Optuna_{model_name}",
                      artifact_path = f'challenger_optuna_{model_name}' if model_name == 'XGB_Classifier' else f'challenger_optuna_{model_name}')

        tuner_report['Optuna'] = {'Cost':find_param.best_value, 'params': find_param.best_params}
        print (f"Optuna: {model_name} --- {tuner_report['Optuna']}\n\n")

####################################################### HYPEROPT #######################################################
    with mlflow.start_run(experiment_id = exp_id,
                          run_name = f"HyperOpt for {model_name}",
                          tags = {"tuner" : "hyperopt",
                                  "run_type": "parent"}) as hyperopt_parent_run:
        parent_run_id = hyperopt_parent_run.info.run_id
        global trial_number
        # trial_number = 0
        def hp_objective(space):
            global trial_number
            trial_number += 1
            with mlflow.start_run(experiment_id = exp_id,
                                  run_name = f"Trial {trial_number} for {model_name} (hyperopt)",
                                  tags = {"run_type": "child"},
                                  nested = True):
                
                if model_name == 'Stacked_Classifier':
                    model = model_class.set_params(**space)
                else:
                    model = model_class(**space)
                # model.set_params(**space)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                cost = eval_metrics(y_test , y_pred)["Cost"]
                print ("Cost: ", cost)

                data = (x_train, y_test, y_pred)
                mlflow_logger(data = data,
                              model = model,
                              model_name = model_name,
                            #   params = model.get_params(),
                              should_log_parent_model = False,
                              artifact_path = f'hyperopt_{model_name}' if model_name == 'XGB_Classifier' else f'hyperopt_{model_name}')
                
                print("Artifacts URI of HyperOpt Child Run: ",mlflow.get_artifact_uri())
                return cost
        print("Artifacts URI of HyperOpt Parent Run: ",mlflow.get_artifact_uri())
        trials = Trials()
        space = {}
        for key,value in params_config['hyperopt'][model_name].items():
            space[key] = eval(value)
        best = fmin(fn= hp_objective,
                    space= space,
                    algo= tpe.suggest,
                    max_evals = 2,
                    trials= trials)
        best_params = space_eval(space,best)

        data = (x_train, x_test, y_train, y_test)
        mlflow_logger(data = data,
                      model_name = model_name,
                      should_log_parent_model = True,
                      run_id = parent_run_id,
                      exp_id = exp_id,
                    #   registered_model_name = f"Challenger_HyperOpt_{model_name}",
                      artifact_path = f'challenger_hyperopt_{model_name}' if model_name == 'XGB_Classifier' else f'challenger_hyperopt_{model_name}')

        tuner_report['HyperOpt'] = {'Cost':int(trials.average_best_error()), 'params': best_params}
        print (f"HyperOpt: {model_name} --- {tuner_report['HyperOpt']}\n\n")
        trial_number = 0

####################################################### Best_COST & Best_Fittable_Params #######################################################
    min_cost_value = min(tuner_report['Optuna']['Cost'],tuner_report['HyperOpt']['Cost'])
    if min_cost_value == tuner_report['Optuna']['Cost']:
        params = tuner_report['Optuna']['params']
    else:
        params = tuner_report['HyperOpt']['params']
    tuner_report['Fittable_Params'] = params
    tuner_report['Best_Cost'] = min_cost_value

    report_[model_name] = tuner_report
    print (f'\n\n{model_name}\nMin Cost: {min_cost_value}\n{report_[model_name]}\n\n')
    # print(report_.values())
    costs = [value['Best_Cost'] for value in report_.values()]
    min_cost = min(costs)
    best_model_so_far_ = [(i, min_cost, report_[i]['Fittable_Params']) for i in report_.keys() if min_cost == report_[i]['Best_Cost']]

    data = x_train
    mlflow_logger(data = data,
                  model_name = model_name,
                #   should_register_model = True,
                  exp_id = exp_id,
                  registered_model_name = f"Challenger_{model_name}",
                  artifact_path = None)
    exp_id_list.append(exp_id)

    return (tuner_report, report_, best_model_so_far_, exp_id_list)

def best_model_finder(report: dict, models: dict):
    best_models_ = sorted(report.items(), key = lambda x: x[1]['Best_Cost'])[:7]
    best_models = [(best_models_[i][0],report[best_models_[i][0]]['Best_Cost']) for i in range(len(best_models_))]
    print('\nBest Models:')
    for i in best_models:
        print(i[0]," Cost: ", i[1],'\n\n')
    best_models_with_params = []
    for i in best_models:
        # print(f"i: {i[0]}")
        best_models_with_params.append((i[0],report[i[0]]['Fittable_Params']))
    best_estimators = {}
    # print("report:\n",report)
    for i in range(len(best_models_with_params)):
        # print ("best_models_with_params[i][0]: \n",best_models_with_params[i][0])
        if (best_models_with_params[i][0] == 'Stacked_Classifier'): #best_models_with_params[i][0] == 'Voting_Classifier'):
            best_estimators[best_models_with_params[i][0]] = models[best_models_with_params[i][0]]
            # best_estimators[best_models_with_params[i][0]].set_params(**best_models_with_params[i][1])

        elif (best_models_with_params[i][0] == 'Voting_Classifier'):
            best_estimators[best_models_with_params[i][0]] = models[best_models_with_params[i][0]]
            # best_estimators[best_models_with_params[i][0]].set_params()
        else:
            best_estimators[best_models_with_params[i][0]] = models[best_models_with_params[i][0]](**best_models_with_params[i][1])
            # best_estimators[best_models_with_params[i][0]].set_params(**best_models_with_params[i][1])
    best_estimators = list(zip(best_estimators.keys(),best_estimators.values()))

    costs = [value['Best_Cost'] for value in report.values()]
    min_cost = min(costs)
    best_model_so_far_ = [(i, min_cost, report[i]['Fittable_Params']) for i in report.keys() if min_cost == report[i]['Best_Cost']]

    return (best_model_so_far_,best_models_with_params,best_estimators)

def stacking_clf_trainer(best_estimators:list[tuple], models: dict, best_model_so_far_: list[tuple],
                         x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame,
                         report: dict):
    stacked_classifier = StackingClassifier(estimators = best_estimators,
                                    final_estimator =  models[best_model_so_far_[0][0]](**best_model_so_far_[0][2]),
                                    cv = 5,
                                    n_jobs = -1,
                                    # passthrough = False,
                                    verbose = 3)
    tuned_params, stacked_clf_report, best_model_so_far, exp_id = parameter_tuning(model_class = stacked_classifier,
                                                                model_name = 'Stacked_Classifier',
                                                                x_train = x_train,
                                                                x_test = x_test,
                                                                y_train = y_train,
                                                                y_test = y_test,
                                                                report_ = report)
    models_names_in_stacking_classifier, models_params_in_stacking_classifier = zip(*best_estimators)
    report['Stacked_Classifier'] = {}
    sc_params = {}
    sc_params['estimators'] = best_estimators
    sc_params['final_estimator'] = stacked_classifier.get_params()['final_estimator']
    sc_params['cv'] = stacked_classifier.get_params()['cv']
    sc_params['n_jobs'] = stacked_classifier.get_params()['n_jobs']

    for i in range(len(models_names_in_stacking_classifier)):
        report['Stacked_Classifier'][models_names_in_stacking_classifier[i]] = models_params_in_stacking_classifier[i].get_params()

    report['Stacked_Classifier']['Best_Cost'] = tuned_params['Best_Cost']

    for i in sc_params.keys():
        tuned_params['Fittable_Params'][i] = sc_params[i]

    report['Stacked_Classifier']['Optuna'] = tuned_params['Optuna']
    report['Stacked_Classifier']['HyperOpt'] = tuned_params['HyperOpt']
    report['Stacked_Classifier']['Fittable_Params'] = tuned_params['Fittable_Params']

    return report,exp_id

    # models['Stacked_Classifier'] = StackingClassifier(**report['Stacked_Classifier']['Fittable_Params'])

def voting_clf_trainer(best_estimators:list[tuple],
                       x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame,
                       report: dict):
    exp_id_voting_clf = mlflow.create_experiment(name = f"61_Voting_Classifier_61",
                                                     tags = {"metrics": "['Balanced_Accuracy_Score', 'F1_Score', 'Accuracy_Score', 'Cost']"})
    with mlflow.start_run(experiment_id = exp_id_voting_clf,
                            run_name = f"Voting_Classifier",
                            tags = {"run_type": "parent"}) as voting_clf_run:
        # voting_clf_run_id = voting_clf_run.info.run_id
        voting_classifier_ = VotingClassifier(estimators = best_estimators,
                                                        voting = "hard",
                                                        weights = None,
                                                        n_jobs = -1,
                                                        verbose = True)
        voting_classifier_.fit(x_train,y_train)
        y_pred = voting_classifier_.predict(x_test)
        cost = eval_metrics(y_true = y_test, y_pred = y_pred)['Cost']
        print(f"\nVoting Classifier Cost: {cost} \n")
        data = (x_train, y_test, y_pred)
        
        mlflow_logger(data = data,
                    model = voting_classifier_,
                    model_name = 'Voting_Classifier',
                    should_log_parent_model = False,
                    artifact_path = f'voting_classifier')
        
        mlflow_logger(data = data,
                    model_name = 'Voting_Classifier',
                    exp_id = exp_id_voting_clf,
                    registered_model_name = f"Challenger_Voting_Classifier",
                    artifact_path = None)
        
        vc_params = {}
        vc_params['estimators'] = best_estimators
        vc_params['voting'] = voting_classifier_.get_params()['voting']
        vc_params['weights'] = voting_classifier_.get_params()['weights']
        vc_params['n_jobs'] = voting_classifier_.get_params()['n_jobs']

        report['Voting_Classifier'] = {}
        report['Voting_Classifier']['Best_Cost'] = cost

        report['Voting_Classifier']['Fittable_Params'] = vc_params

    return report,exp_id_voting_clf

def model_trainer(x_train : pd.DataFrame, y_train : pd.DataFrame, x_test : pd.DataFrame, y_test : pd.DataFrame,
                  model_: ML_Model = None, 
                  models: dict = None,
                  params: dict = None,
                  best_model_details: list[tuple] = None):
    if models:
        model_class = models[best_model_details[0][0]]
        print(f"\nBest Model Details: {best_model_details}")
        print("\nModel: ",model_class)
        print("\nModels: ",models)
        print(f"\nModel params: {best_model_details[0][2]}")
        model = model_class(**best_model_details[0][2])
    elif model:
        model = model_(**params)
        print("\nModel: ",model)
    model.fit(x_train, y_train)
    print(f"\nmodel_getparams(): {model.get_params()}\n")
    y_pred = model.predict(x_test)
    cost = eval_metrics(y_true = y_test, y_pred = y_pred)['Cost']
    
    return cost

def mlflow_logger(artifact_path: str, data = None, model = None, model_name: str = None, 
                  should_log_parent_model: bool = False, should_register_champion_model:bool = False, registered_model_name: str = None, 
                  run_id: str =  None, exp_id: int|list = None):
    if not artifact_path and should_register_champion_model == False:
        # x_train = data
        print("Client_Tracking_URI: ", client.tracking_uri)
        print("Client_Registry_URI: ", client._registry_uri)
        filter_string = f"tags.run_type ilike 'parent'"
        best_run_id = mlflow.search_runs(experiment_ids=[exp_id],
                                         order_by = ['metrics.Cost'],
                                         filter_string = filter_string)[['run_id','artifact_uri','metrics.Cost']]['run_id'][0]
        best_artifact_path = mlflow.search_runs(experiment_ids=[exp_id],
                                                order_by = ['metrics.Cost'],
                                                filter_string = filter_string)[['run_id','artifact_uri','metrics.Cost']]['artifact_uri'][0]
        artifact_path_name = client.list_artifacts(f'{best_run_id}')[0].path
        print(f"\nBest_Run_ID: {best_run_id}")
        print(f"Best_Model's_Artifact_Path: {best_artifact_path}/{artifact_path_name}")

        client.create_registered_model(name = registered_model_name)
        client.create_model_version(name = registered_model_name,
                                    source = f"{best_artifact_path}/{artifact_path_name}",
                                    run_id = best_run_id)
    
    elif not artifact_path and should_register_champion_model == True:
        parent_runs = mlflow.search_registered_models()
        print("Experiment IDs: ",exp_id)
        runs_df = mlflow.search_runs(experiment_ids = exp_id,
                            search_all_experiments = True,
                            filter_string = f"tags.run_type ilike 'parent'")
        runs_list_ = [parent_runs[i].latest_versions[0].run_id for i in range(len(parent_runs))]
        best_run = runs_df[runs_df['run_id'].isin(runs_list_)].sort_values(by = "metrics.Cost").reset_index(drop=True)['run_id'][0]
        best_artifact = runs_df[runs_df['run_id'].isin(runs_list_)].sort_values(by = "metrics.Cost").reset_index(drop=True)['artifact_uri'][0]
        artifact_path_name = client.list_artifacts(f'{best_run}')[0].path
        model_name = runs_df[runs_df['run_id'].isin(runs_list_)].sort_values(by = "metrics.Cost").reset_index(drop=True)['tags.mlflow.runName'][0]
        model_name = model_name.replace("HyperOpt for ", "").replace("Optuna for ", "")
        client.create_registered_model(name = f"Champion {model_name}",
                                    tags = {"model_type": "champion"},
                                    description = f"{model_name} is the new champion model")
        client.create_model_version(name = f"Champion {model_name}",
                                    source = f"{best_artifact}/{artifact_path_name}",
                                    run_id = best_run,
                                    tags = {"model_type" : "champion",
                                            "model_name" : model_name})

    elif should_log_parent_model == True and should_register_champion_model == False:
        x_train, x_test, y_train, y_test = data
        print("Experiment IDs: ",exp_id)
        filter_string=f"tags.mlflow.parentRunId ILIKE '{run_id}'"
        best_run_id = mlflow.search_runs(experiment_ids=[exp_id],
                        filter_string=filter_string,
                        order_by = ['metrics.Cost'])[['run_id','artifact_uri','metrics.Cost']]['run_id'][0]
        best_artifact_path = mlflow.search_runs(experiment_ids=[exp_id],
                        filter_string=filter_string,
                        order_by = ['metrics.Cost'])[['run_id','artifact_uri','metrics.Cost']]['artifact_uri'][0]
        artifact_path_name = client.list_artifacts(f'{best_run_id}')[0].path
        print(f"Parent_Run_ID: {run_id}")
        print(f"Artifact_Path: {best_artifact_path}/{artifact_path_name}")
        if model_name == 'XGB_Classifier':
            best_model = mlflow.xgboost.load_model(f"{best_artifact_path}/{artifact_path_name}")
            params = client.get_run(best_run_id).data.params
            for key,value in params.items():
                try:
                    params[key] = eval(value)
                except:
                    params[key] = value
                    if value == 'nan':
                        params[key] = np.nan
            print("Best Params:\n",{key: value for key, value in params.items() if value is not None},"\n")
            signature = mlflow.xgboost.infer_signature(model_input = x_train,
                                                        model_output = best_model.predict(x_train),
                                                        params = {key: value for key, value in params.items() if value is not None})
            mlflow.xgboost.log_model(xgb_model = best_model,
                                     artifact_path = artifact_path,
                                     signature = signature)
        else:
            best_model = mlflow.sklearn.load_model(f"{best_artifact_path}/{artifact_path_name}")
            params = client.get_run(best_run_id).data.params
            if model_name == "Stacked_Classifier" or model_name == "voting_classifier":
                signature = mlflow.models.infer_signature(model_input = x_train,
                                                        model_output = best_model.predict(x_train),
                                                        params = params)
                mlflow.sklearn.log_model(sk_model = best_model,
                                        artifact_path = artifact_path, 
                                        signature = signature)
                for key,value in params.items():
                    try:
                        params[key] = eval(value)
                    except:
                        params[key] = value
                        if value == 'nan':
                            params[key] = np.nan
                params = {key: value for key, value in params.items() if value is not None}
            else:
                for key,value in params.items():
                    try:
                        params[key] = eval(value)
                    except:
                        params[key] = value
                        if value == 'nan':
                            params[key] = np.nan
                signature = mlflow.models.infer_signature(model_input = x_train,
                                                            model_output = best_model.predict(x_train),
                                                            params = {key: value for key, value in params.items() if value is not None})
                mlflow.sklearn.log_model(sk_model = best_model,
                                            artifact_path = artifact_path, 
                                            signature = signature)
        print("Best Params:\n",{key: value for key, value in params.items() if value is not None},"\n")
        if model_name == "Stacked_Classifier" or model_name == 'Voting_Classifier':
            flag = 0
            for i in range(len(best_model.get_params()['estimators'])):
                if 'XGB_Classifier' == best_model.get_params()['estimators'][i][0]:
                    flag =1
            if flag == 1:
                clf_list = []
                for i in range(len(best_model.get_params()['estimators'])):
                    if best_model.get_params()['estimators'][i][0] != "XGB_Classifier":
                        clf_list.append((best_model.get_params()['estimators'][i][0],best_model.get_params()['estimators'][i][1].__class__()))
                
                estimators_params = {}
                for i in range(len(best_model.get_params()['estimators'])):
                    estimators_params[best_model.get_params()['estimators'][i][0]] = best_model.get_params()['estimators'][i][1].get_params()
                for i in estimators_params:
                    estimators_params[i] = {key:value for key,value in estimators_params[i].items() if value is not None}
                
                if model_name == "Stacked_Classifier":
                    if best_model.get_params()['final_estimator'].__class__.__name__ == 'XGBClassifier':
                            s_clf_params = {'estimators' : clf_list,
                                            'final_estimator' : 'XGBClassifier()',
                                            'cv' : best_model.get_params()['cv'],
                                            'stack_method' : best_model.get_params()['stack_method'],
                                            'passthrough' : best_model.get_params()['passthrough']}
                            for key,value in estimators_params.items():
                                s_clf_params[f"{key}_Params"] = value
                    else:
                        s_clf_params = {'estimators' : clf_list,
                                'final_estimator' : best_model.get_params()['final_estimator'],
                                'cv' : best_model.get_params()['cv'],
                                'stack_method' : best_model.get_params()['stack_method'],
                                'passthrough' : best_model.get_params()['passthrough']}
                        for key,value in estimators_params.items():
                                s_clf_params[f"{key}_Params"] = value

                    print("Processed_NEW_S_CLF_Params: ",s_clf_params)
                    mlflow.log_params(params = s_clf_params)

                elif model_name == "Voting_Classifier":
                    v_clf_params = {'estimators' : clf_list}
                    for key,value in estimators_params.items():
                        v_clf_params[f"{key}_Params"] = value
                    print("Processed_NEW_V_CLF_Params: ",v_clf_params)
                    mlflow.log_params(params = v_clf_params)

            else:
                clf_list = []
                for i in range(len(best_model.get_params()['estimators'])):
                    clf_list.append((best_model.get_params()['estimators'][i][0],best_model.get_params()['estimators'][i][1].__class__()))
                
                estimators_params = {}
                for i in range(len(best_model.get_params()['estimators'])):
                        estimators_params[best_model.get_params()['estimators'][i][0]] = best_model.get_params()['estimators'][i][1].get_params()
                for i in estimators_params:
                        estimators_params[i] = {key:value for key,value in estimators_params[i].items() if value is not None}

                if model_name == "Stacked_Classifier":
                    s_clf_params = {'estimators' : clf_list,
                                    'final_estimator' : best_model.get_params()['final_estimator'],
                                    'cv' : best_model.get_params()['cv'],
                                    'stack_method' : best_model.get_params()['stack_method'],
                                    'passthrough' : best_model.get_params()['passthrough']}
                    for key,value in estimators_params.items():
                            s_clf_params[f"{key}_params"] = value
                    print("Processed_NEW_S_CLF_Params: ",s_clf_params)
                    mlflow.log_params(params = s_clf_params)
                    
                elif model_name == "Voting_Classifier":
                    v_clf_params = {'estimators' : clf_list}
                    for key,value in estimators_params.items():
                        v_clf_params[f"{key}_Params"] = value
                    print("Processed_NEW_V_CLF_Params: ",v_clf_params)
                    mlflow.log_params(params = v_clf_params)

                # new_params = best_model.get_params()
                # for key,value in new_params.items():
                #     try:
                #         new_params[key] = eval(value)
                #     except:
                #         new_params[key] = str(value)
                #         if value == 'nan':
                #             new_params[key] = np.nan
                # processed_new_params = {key: value for key, value in new_params.items() if value is not None}
                # print("Processed_NEW_Params: ",processed_new_params)
                # mlflow.log_params(params = processed_new_params)
        else:
            if model_name == 'XGB_Classifier':
                params = client.get_run(best_run_id).data.params
                mlflow.log_params(params = params)
            else:
                mlflow.log_params(params = best_model.get_params())

        # if model_name == "Stacked_Classifier" or model_name == 'Voting_Classifier':
        mlflow.log_metrics(metrics = eval_metrics(y_test , best_model.fit(x_train, y_train).predict(x_test)))
        # else:
            # mlflow.log_metrics(metrics = eval_metrics(y_test , best_model.set_params(**params).fit(x_train, y_train).predict(x_test)))

    else:
        x_train, y_test, y_pred = data
        mlflow.log_metrics(metrics = eval_metrics(y_test , y_pred)) 
        if model_name == "Stacked_Classifier" or model_name == 'Voting_Classifier':
            flag = 0
            for i in range(len(model.get_params()['estimators'])):
                if 'XGB_Classifier' == model.get_params()['estimators'][i][0]:
                    flag =1
            if flag == 1:
                clf_list = []
                for i in range(len(model.get_params()['estimators'])):
                    if model.get_params()['estimators'][i][0] != "XGB_Classifier":
                        clf_list.append((model.get_params()['estimators'][i][0],model.get_params()['estimators'][i][1].__class__()))
                
                estimators_params = {}
                for i in range(len(model.get_params()['estimators'])):
                    estimators_params[model.get_params()['estimators'][i][0]] = model.get_params()['estimators'][i][1].get_params()
                for i in estimators_params:
                    estimators_params[i] = {key:value for key,value in estimators_params[i].items() if value is not None}
                
                if model_name == "Stacked_Classifier":
                    if model.get_params()['final_estimator'].__class__.__name__ == 'XGBClassifier':
                            s_clf_params = {'estimators' : clf_list,
                                            'final_estimator' : 'XGBClassifier()',
                                            'cv' : model.get_params()['cv'],
                                            'stack_method' : model.get_params()['stack_method'],
                                            'passthrough' : model.get_params()['passthrough']}
                            for key,value in estimators_params.items():
                                s_clf_params[f"{key}_Params"] = value
                    else:
                        s_clf_params = {'estimators' : clf_list,
                                'final_estimator' : model.get_params()['final_estimator'],
                                'cv' : model.get_params()['cv'],
                                'stack_method' : model.get_params()['stack_method'],
                                'passthrough' : model.get_params()['passthrough']}
                        for key,value in estimators_params.items():
                                s_clf_params[f"{key}_Params"] = value

                    print("Processed_NEW_S_CLF_Params: ",s_clf_params)
                    mlflow.log_params(params = s_clf_params)

                elif model_name == "Voting_Classifier":
                    v_clf_params = {'estimators' : clf_list}
                    for key,value in estimators_params.items():
                        v_clf_params[f"{key}_Params"] = value
                    print("Processed_NEW_V_CLF_Params: ",v_clf_params)
                    mlflow.log_params(params = v_clf_params)

            else:
                clf_list = []
                for i in range(len(model.get_params()['estimators'])):
                    clf_list.append((model.get_params()['estimators'][i][0],model.get_params()['estimators'][i][1].__class__()))
                
                estimators_params = {}
                for i in range(len(model.get_params()['estimators'])):
                        estimators_params[model.get_params()['estimators'][i][0]] = model.get_params()['estimators'][i][1].get_params()
                for i in estimators_params:
                        estimators_params[i] = {key:value for key,value in estimators_params[i].items() if value is not None}

                if model_name == "Stacked_Classifier":
                    s_clf_params = {'estimators' : clf_list,
                                    'final_estimator' : model.get_params()['final_estimator'],
                                    'cv' : model.get_params()['cv'],
                                    'stack_method' : model.get_params()['stack_method'],
                                    'passthrough' : model.get_params()['passthrough']}
                    for key,value in estimators_params.items():
                            s_clf_params[f"{key}_params"] = value
                    print("Processed_NEW_S_CLF_Params: ",s_clf_params)
                    mlflow.log_params(params = s_clf_params)
                    
                elif model_name == "Voting_Classifier":
                    v_clf_params = {'estimators' : clf_list}
                    for key,value in estimators_params.items():
                        v_clf_params[f"{key}_Params"] = value
                    print("Processed_NEW_V_CLF_Params: ",v_clf_params)
                    mlflow.log_params(params = v_clf_params)
        else:
            mlflow.log_params(params = model.get_params())
        if model_name == 'XGB_Classifier': 
            signature = mlflow.xgboost.infer_signature(model_input = x_train,
                                                      model_output = model.predict(x_train),
                                                      params = {key: value for key, value in model.get_params().items() if value is not None}) 
            mlflow.xgboost.log_model(xgb_model = model, 
                                        artifact_path = artifact_path,
                                        signature = signature)
        elif model_name == "Stacked_Classifier" or model_name == 'Voting_Classifier':     
            params = model.get_params()
            for key, value in params.items():
                params[key] = str(value)
            signature = mlflow.models.infer_signature(model_input = x_train,
                                                      model_output = model.predict(x_train),
                                                      params = params) 
            mlflow.sklearn.log_model(sk_model = model, 
                                        artifact_path = artifact_path,
                                        signature = signature)   
            
        else:
            signature = mlflow.models.infer_signature(model_input = x_train,
                                                      model_output = model.predict(x_train),
                                                      params = {key: value for key, value in model.get_params().items() if value is not None}) 
            mlflow.sklearn.log_model(sk_model = model, 
                                        artifact_path = artifact_path,
                                        signature = signature)   
