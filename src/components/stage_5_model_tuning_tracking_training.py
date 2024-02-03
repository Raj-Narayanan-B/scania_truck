import pandas as pd
import os
import mlflow
import re #type: ignore
# exp_count = 276
from src.utils import (parameter_tuning_2, mlflow_logger, model_trainer, best_model_finder,
                       params_evaluator, eval_metrics, save_yaml, voting_clf_trainer, stacking_clf_trainer)

from src.constants import *
from src.components.stage_3_data_split import data_splitting_component
from src.components.stage_4_final_preprocessing import stage_4_final_processing_component
from src.config.configuration_manager import ConfigurationManager
from src.entity.entity_config import (Stage2ProcessingConf,
                                      ModelMetricsConf, 
                                      ModelTrainerConf, 
                                      PreprocessorConf, 
                                      DataSplitConf,
                                      Stage1ProcessingConf)
from src import logger

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, 
                              HistGradientBoostingClassifier, StackingClassifier, VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

import mlflow.pyfunc
from mlflow.client import MlflowClient

class model_tuning_tracking_component:
    def __init__(self,
                 stage_2_conf: Stage2ProcessingConf,
                 metrics_conf: ModelMetricsConf,
                 model_conf: ModelTrainerConf,
                 preprocessor_conf: PreprocessorConf,
                 data_split_conf: DataSplitConf,
                 stage1_processor_conf: Stage1ProcessingConf) -> None:
        self.stage_2_config = stage_2_conf
        self.metrics_config = metrics_conf
        self.preprocessor_config = preprocessor_conf
        self.model_config = model_conf
        self.split_config = data_split_conf
        self.stage1_processor_config = stage1_processor_conf

    def models_tuning (self):
        logger.info("loading training and testing datasets")
        # size = 2000 #################CHANGED

        # stage_3_data_split_obj = data_splitting_component(data_split_conf = self.split_config,
        #                                                   stage1_processor_conf = self.stage1_processor_config)
        # train_data_training_set,train_data_testing_set = stage_3_data_split_obj.data_splitting(size)#################CHANGED

        # stage_4_final_processing_obj = stage_4_final_processing_component(data_split_conf = self.split_config,
        #                                                                   stage_2_processor_conf = self.stage_2_config,
        #                                                                   preprocessor_conf = self.preprocessor_config)
        # train_df, test_df = stage_4_final_processing_obj.final_processing(train_data_training_set,train_data_testing_set) #################CHANGED

        train_df = pd.read_csv(self.stage_2_config.train_data_path)
        test_df = pd.read_csv(self.stage_2_config.test_data_path)   #################CHANGED
        
        # Initialize x_train, x_test, y_train, y_test
        x_train, y_train = train_df.drop(columns = 'class'), train_df['class']
        x_test, y_test = test_df.drop(columns = 'class'), test_df['class']

        client = MlflowClient(tracking_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow",
                      registry_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow")
        
        dataframe = pd.read_csv(self.stage1_processor_config.train_data_path)

        models = {'Logistic_Regression': LogisticRegression, 
                  'SGD_Classifier': SGDClassifier,
                  'Random Forest': RandomForestClassifier, 
                  'Ada_Boost': AdaBoostClassifier, 
                  'Grad_Boost': GradientBoostingClassifier, 
                  'Bagging_Classifier': BaggingClassifier, 
                  'ExtraTreesClassifier': ExtraTreesClassifier, 
                  'Hist_Grad_Boost_Classifier': HistGradientBoostingClassifier, 
                  'Decision_Tree_Classifier': DecisionTreeClassifier,
                  'XGB_Classifier': XGBClassifier,
                  'Light_GBM' : LGBMClassifier,
                  'KNN_Classifier': KNeighborsClassifier,
                  }
        logger.info("Commencing models' Hyper_Parameter Tuning")
        # optuna_study_df, exp_id_list, model_accuracies, best_exp_id = parameter_tuning_2(models = models,  #################CHANGED
        #                                                                     client = client,
        #                                                                     dataframe = dataframe)
        logger.info("Hyper_Parameter Tuning complete")
        best_exp_id = '574'  #################CHANGED
        best_exp_ids = []
        best_exp_ids.append(best_exp_id)
        
        # optuna_study_df = pd.read_csv("F:\iNeuron\Projects\scania_failures_2\\artifacts\metrics\model_trial_study_df.csv")
        
        # # Get the dictionary of all the registered challenger models from MLFlow.  
        # # This dict will have model names as keys and the run_IDs as values.
        # registered_models = {mlflow.search_registered_models()[i].latest_versions[0].name : mlflow.search_registered_models()[i].latest_versions[0].run_id for i in range(len(mlflow.search_registered_models()))}

        # # Get the accuracies and the respective params of the models as dict
        # run_details = {}
        # for key,value in registered_models.items():
        #     run_details[client.get_run(value).data.tags['model']] = {}
        #     run_details[client.get_run(value).data.tags['model']]['accuracy'] = client.get_run(value).data.metrics['Accuracy_Score']
        #     run_details[client.get_run(value).data.tags['model']]['params'] = params_evaluator(client.get_run(value).data.params)

        # # Create a dataframe from the "run_details" dict and sort it by "accuracy" in DESC
        # # In this dataframe only the models whose accuracy is greater than 0.9 are chosen.
        # models_df = pd.DataFrame(run_details).T
        
        # sorted_models_df = models_df[models_df['accuracy'] > 0.9].sort_values(by = 'accuracy', ascending=False)

        # # Using the sorted_models_df from above, we are creating another dict that has the models fitted with the parameters.
        # mlflow_models = {key:value(**(sorted_models_df.params[key])) for key,value in models.items() if  key in sorted_models_df.index}

        # # Create the list[tuple] best_estimators to fit in the voting classifier
        # best_estimators_mlflow = list(zip(mlflow_models.keys(),mlflow_models.values()))

        # print(f"\nBest_Estimators: {best_estimators_mlflow}")

        # # If using stacking classifier, get the final estimator using:
        # final_estimator_mlflow = {key:value(**(sorted_models_df.iloc[:1,:].params[key])) for key,value in models.items() if key in sorted_models_df.iloc[:1,:].index}
        # # Access the final estimator model using:

        # final_estimator = final_estimator_mlflow[sorted_models_df.iloc[:1,:].index.values[0]]
        best_estimators_mlflow, final_estimator_mlflow, final_estimator = best_model_finder(models = models, client = client)
        print(f"\nBest_Estimators: {best_estimators_mlflow}\n")
        print(f"\nFinal_estimator: {final_estimator_mlflow}\n")

        # ## Fitting the final estimator on the entire data:
        # final_estimator.fit(X = x_train, y = y_train)
        # y_pred_final_estimator = final_estimator.predict(X = x_test)
        # metrics_final_estimator = eval_metrics(y_true = y_test, y_pred = y_pred_final_estimator)
        # metrics_final_estimator['model'] = final_estimator
        # metrics_final_estimator['model_name'] = list(final_estimator_mlflow.keys())[0]

        metrics_final_estimator, y_pred_final_estimator, final_estimator_run_id = model_trainer(x_train = x_train, y_train = y_train,
                                                                        x_test = x_test, y_test = y_test,
                                                                        model = final_estimator, 
                                                                        model_dict = final_estimator_mlflow,
                                                                        mlflow_experiment_id = best_exp_id,
                                                                        client = client)

        metrics_stacking_clf,exp_id_stack_clf, y_pred_stacking_clf, stacking_clf_run_id = stacking_clf_trainer(best_estimators = best_estimators_mlflow,
                                                                                          final_estimator = final_estimator,
                                                                                          x_train = x_train, y_train = y_train,
                                                                                          x_test = x_test, y_test = y_test,
                                                                                          client = client)
        best_exp_ids.append(exp_id_stack_clf)

        # # Create the stacking_classifier object
        # stacking_clf = StackingClassifier(estimators = best_estimators_mlflow,
        #                                   final_estimator = final_estimator,
        #                                   cv = 3,
        #                                   n_jobs = -1,
        #                                   verbose = 3,
        #                                   passthrough = True)
        
        # # Test with Stacking_classifier
        # stacking_clf.fit(X = x_train, y = y_train)
        # y_pred_stacking_clf = stacking_clf.predict(X = x_test)
        # metrics_stacking_clf = eval_metrics(y_true = y_test, y_pred = y_pred_stacking_clf)
        # metrics_stacking_clf['model'] = stacking_clf
        # metrics_stacking_clf['model_name'] = 'Stacked_Classifier'

        # # Log Stacking_CLF in MLFlow
        # tags = {"metrics": "['Balanced_Accuracy_Score', 'F1_Score', 'Accuracy_Score', 'Cost']"}
        # exp_id_stack_clf = mlflow.create_experiment(name = f"{exp_count}_Stacked_Classifier_{exp_count}", tags = tags)
        # with mlflow.start_run(experiment_id = exp_id_stack_clf,
        #                       run_name = f"Challenger {stacking_clf.__class__.__name__}",
        #                       tags = {'model' : 'Stacked_Classifier',
        #                               "run_type": "parent",
        #                               "model_type" : "Challenger"}) as parent_run:
        #     parent_run_id = parent_run.info.run_id
        #     mlflow_logger(model = stacking_clf,
        #                   client = client, 
        #                   model_name = 'Stacked_Classifier',
        #                   should_log_parent_model = False,
        #                   artifact_path = f'candidate_Stacked_Classifier')
            
        #     mlflow.log_metrics(metrics = {"Accuracy_Score": metrics_stacking_clf['Accuracy_Score']})  

        #     mlflow_logger(client = client,
        #                 model_name = 'Stacked_Classifier',
        #                 exp_id = exp_id_stack_clf,
        #                 registered_model_name = f"Challenger Stacked_Classifier",
        #                 artifact_path = None)
        # exp_id_list.append(exp_id_stack_clf)    #################CHANGED
        # best_exp_ids.append(exp_id_stack_clf)

        # Create the voting_classifier object
        # voting_clf = VotingClassifier(estimators=best_estimators_mlflow,
        #                       n_jobs = -1,
        #                       verbose = True)
        
        #CALL, TRAIN, LOG VOTING CLASSIFIER
        metrics_voting_clf, exp_id_voting_clf, y_pred_voting_clf, voting_clf_run_id = voting_clf_trainer(best_estimators = best_estimators_mlflow, 
                                                                                      x_train = x_train, y_train = y_train,
                                                                                      x_test = x_test, y_test = y_test,
                                                                                      client = client)
        best_exp_ids.append(exp_id_voting_clf)

        # # Test with Voting_Classifier
        # voting_clf.fit(X = x_train, y = y_train)
        # y_pred_voting_clf = voting_clf.predict(X = x_test)
        # metrics_voting_clf = eval_metrics(y_true = y_test, y_pred = y_pred_voting_clf)
        # metrics_voting_clf['model'] = voting_clf
        # metrics_voting_clf['model_name'] = 'Voting_Classifier'

        # # Log Voting_CLF in MLFlow
        # exp_id_voting_clf = mlflow.create_experiment(name = f"{exp_count}_Voting_Classifier_{exp_count}", tags = tags)
        # with mlflow.start_run(experiment_id = exp_id_voting_clf,
        #                       run_name = f"Challenger {voting_clf.__class__.__name__}",
        #                       tags = {'model' : 'Voting_Classifier',
        #                               "run_type": "parent",
        #                               "model_type" : "Challenger"}) as parent_run:
        #     parent_run_id = parent_run.info.run_id
        #     mlflow_logger(model = voting_clf,
        #                   client = client, 
        #                   model_name = 'Voting_Classifier',
        #                   should_log_parent_model = False,
        #                   artifact_path = f'candidate_Voting_Classifier')
            
        #     mlflow.log_metrics(metrics = {"Accuracy_Score": metrics_voting_clf['Accuracy_Score']})

        #     mlflow_logger(client = client,
        #                   model_name = 'Voting_Classifier',
        #                   exp_id = exp_id_voting_clf,
        #                   registered_model_name = f"Challenger Voting_Classifier",
        #                   artifact_path = None)
        # exp_id_list.append(exp_id_voting_clf)           #################CHANGED
        # best_exp_ids.append(exp_id_voting_clf)

        best_accuracy = max([metrics_voting_clf['Accuracy_Score'],metrics_stacking_clf['Accuracy_Score'],metrics_final_estimator['Accuracy_Score']])
        
        print(f"\nMetrics are: \n***************************\nStacking_CLF:\n{metrics_stacking_clf['Accuracy_Score']}\n**************************\nVoting_CLF:\n{metrics_voting_clf['Accuracy_Score']}\n**************************\n{metrics_final_estimator['model'].__class__.__name__}:\n{metrics_final_estimator['Accuracy_Score']}\n**************************\n")

        # with mlflow.start_run(experiment_id = best_exp_id,
        #                         run_name = f"Challenger {metrics_final_estimator['model'].__class__.__name__}",
        #                         tags = {'model' : f"{metrics_final_estimator['model'].__class__.__name__}",
        #                             "run_type": "parent",
        #                             "model_type" : "Challenger"}) as final_estimator_run:
        #     mlflow.sklearn.log_model(sk_model = metrics_final_estimator['model'], 
        #                             artifact_path = f"challenger_{metrics_final_estimator['model'].__class__.__name__}",
        #                             registered_model_name = f"Challenger {metrics_final_estimator['model'].__class__.__name__}")
        #     mlflow.log_metrics(eval_metrics(y_true = y_test, y_pred = y_pred_final_estimator))

        # if best_accuracy == metrics_voting_clf['Accuracy_Score']:
        #     champion_model_name = metrics_voting_clf['model'].__class__.__name__
        #     print(f"\n\nChampion model is: {champion_model_name}\n")
        #     print(f"\nMetrics are: \n{metrics_voting_clf}\n\n")
        #     client.set_registered_model_tag(name = 'Challenger Stacked_Classifier',
        #                                     key = 'model_type',
        #                                     value = 'Challenger')
        #     client.set_registered_model_tag(name = f"Challenger {metrics_final_estimator['model'].__class__.__name__}",
        #                                     key = 'model_type',
        #                                     value = 'Challenger')
        #     # mlflow_logger(exp_id = [exp_id_voting_clf],
        #     #     should_register_model=True,
        #     #     client = client,
        #     #     is_tuning_complete = True,
        #     #     registered_model_name = 'Champion',
        #     #     artifact_path=None)

        # elif best_accuracy == metrics_stacking_clf['Accuracy_Score']:
        #     champion_model_name = metrics_stacking_clf['model'].__class__.__name__
        #     print(f"\n\nChampion model is: {champion_model_name}\n")
        #     print(f"\nMetrics are: \n{metrics_stacking_clf}\n\n")
        #     client.set_registered_model_tag(name = f"Challenger {metrics_final_estimator['model'].__class__.__name__}",
        #                                     key = 'model_type',
        #                                     value = 'Challenger')
        #     client.set_registered_model_tag(name = 'Challenger Voting_Classifier',
        #                                     key = 'model_type',
        #                                     value = 'Challenger')
        #     # mlflow_logger(exp_id = [exp_id_stack_clf],
        #     #     should_register_model=True,
        #     #     client = client,
        #     #     is_tuning_complete = True,
        #     #     registered_model_name = 'Champion',
        #     #     artifact_path=None)

        # else:
        #     champion_model_name = metrics_final_estimator['model'].__class__.__name__
        #     print(f"\n\nChampion model is: {champion_model_name}\n")
        #     print(f"\nMetrics are: \n{metrics_final_estimator}\n\n")

        # client.set_registered_model_tag(name = f"Challenger {metrics_final_estimator['model'].__class__.__name__}",
        #                                 key = 'model_type',
        #                                 value = 'Challenger')
        # client.set_registered_model_tag(name = 'Challenger Voting_Classifier',
        #                                 key = 'model_type',
        #                                 value = 'Challenger')
        # client.set_registered_model_tag(name = 'Challenger Stacked_Classifier',
        #                                 key = 'model_type',
        #                                 value = 'Challenger')

        # To register the champion model
        # mlflow_logger(exp_id = best_exp_ids,
        #               should_register_model=True,
        #               client = client,
        #               is_tuning_complete = True,
        #               registered_model_name = 'Champion',
        #               artifact_path=None)
     
        report = {}
        report['Stacking_Classifier'] = eval_metrics(y_true = y_test, y_pred = y_pred_stacking_clf)
        report['Voting_Classifier'] = eval_metrics(y_true = y_test, y_pred = y_pred_voting_clf)
        report[f"{metrics_final_estimator['model'].__class__.__name__}"] = eval_metrics(y_true = y_test, y_pred = y_pred_final_estimator)
        
        # save_yaml(file = artifact_path_dict, filepath = f"{self.model_config.root_dir}/artifact_path.yaml")
        # # save_yaml(file = model.metadata.artifact_path, filepath = self.model_config.root_dir)

        logger.info(f"Saving metrics.yaml file at {self.metrics_config.metrics}")
        save_yaml(file = report, filepath = self.metrics_config.metrics)

        # logger.info(f"Saving best_metrics.yaml file at {self.metrics_config.best_metric}")
        # save_yaml(file = {best_model_sofar[0][0]: report[best_model_sofar[0][0]]}, filepath = self.metrics_config.best_metric)

        # logger.info(f"Saving model at {self.model_config.model_path}")
        # save_binary(file = models[best_model_sofar[0][0]],filepath = self.model_config.model_path)


# conf_obj = ConfigurationManager()
# stage_2_obj = conf_obj.get_stage2_processing_config()
# model_metrics_obj = conf_obj.get_metric_config()
# model_config_obj = conf_obj.get_model_config()
# data_split_obj = conf_obj.get_data_split_config()
# preprocessor_obj = conf_obj.get_preprocessor_config()
# stage_1_obj = conf_obj.get_stage1_processing_config()

# obj = model_tuning_tracking_component(stage_2_conf = stage_2_obj,
#                                       metrics_conf = model_metrics_obj,
#                                       model_conf = model_config_obj,
#                                       preprocessor_conf = preprocessor_obj,
#                                       data_split_conf = data_split_obj,
#                                       stage1_processor_conf = stage_1_obj)
# obj.models_tuning()