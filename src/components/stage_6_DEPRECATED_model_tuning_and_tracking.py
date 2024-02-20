# import os
# import mlflow

# from src.utils import (load_yaml, save_yaml, save_binary,
#                        eval_metrics, parameter_tuning, best_model_finder,
#                        stacking_clf_trainer, voting_clf_trainer, model_trainer, mlflow_logger)
# from src.constants import SCHEMA_PATH
# from src.components.stage_3_data_split import data_splitting_component
# from src.components.stage_4_final_preprocessing import stage_4_final_processing_component
# from src.config.configuration_manager import ConfigurationManager
# from src.entity.entity_config import (Stage2ProcessingConf,
#                                       ModelMetricsConf,
#                                       ModelTrainerConf,
#                                       PreprocessorConf,
#                                       DataSplitConf,
#                                       Stage1ProcessingConf)
# from src import logger

# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
#                               GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier,
#                               HistGradientBoostingClassifier, StackingClassifier, VotingClassifier)
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# # from catboost import CatBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier

# import mlflow.pyfunc
# from mlflow.client import MlflowClient
# client = MlflowClient(tracking_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow",
#                       registry_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow")


# class model_tuning_tracking_component:
#     def __init__(self,
#                  stage_2_conf: Stage2ProcessingConf,
#                  metrics_conf: ModelMetricsConf,
#                  model_conf: ModelTrainerConf,
#                  preprocessor_conf: PreprocessorConf,
#                  data_split_conf: DataSplitConf,
#                  stage1_processor_conf: Stage1ProcessingConf) -> None:
#         self.stage_2_config = stage_2_conf
#         self.metrics_config = metrics_conf
#         self.preprocessor_config = preprocessor_conf
#         self.model_config = model_conf
#         self.split_config = data_split_conf
#         self.stage1_processor_config = stage1_processor_conf

#     def models_tuning(self):
#         schema = load_yaml(SCHEMA_PATH)
#         target = list(schema.Target.keys())[0]
#         size = 20000
#         logger.info("loading training and testing datasets")

#         if os.path.exists(self.preprocessor_config.preprocessor_path):
#             os.remove(self.preprocessor_config.preprocessor_path)

#         stage_3_data_split_obj = data_splitting_component(data_split_conf=self.split_config,
#                                                           stage1_processor_conf=self.stage1_processor_config)
#         pre_train_df, pre_test_df = stage_3_data_split_obj.data_splitting(size)

#         stage_4_final_processing_obj = stage_4_final_processing_component(data_split_conf=self.split_config,
#                                                                           stage_2_processor_conf=self.stage_2_config,
#                                                                           preprocessor_conf=self.preprocessor_config)
#         train_df, test_df = stage_4_final_processing_obj.final_processing(pre_train_df, pre_test_df)

#         # train_df = pd.read_csv(self.stage_2_config.train_data_path)
#         # test_df = pd.read_csv(self.stage_2_config.test_data_path)

#         print("Train data's shape: ", train_df.shape)
#         print("Test data's shape: ", test_df.shape)

#         print("\nTrain data's target value_counts: ", train_df[target].value_counts())
#         print("\nTest data's target value_counts: ", test_df[target].value_counts(), '\n')

#         x_train = train_df.drop(columns=target)
#         y_train = train_df[target]

#         x_test = test_df.drop(columns=target)
#         y_test = test_df[target]

#         models = {'Logistic_Regression': LogisticRegression,
#                   'SGD_Classifier': SGDClassifier,
#                   'Random Forest': RandomForestClassifier,
#                   'Ada_Boost': AdaBoostClassifier,
#                   'Grad_Boost': GradientBoostingClassifier,
#                   'Bagging_Classifier': BaggingClassifier,
#                   'ExtraTreesClassifier': ExtraTreesClassifier,
#                   'Hist_Grad_Boost_Classifier': HistGradientBoostingClassifier,
#                   'Decision_Tree_Classifier': DecisionTreeClassifier,
#                   'XGB_Classifier': XGBClassifier,
#                   'KNN_Classifier': KNeighborsClassifier,
#                   }
#         logger.info("Commencing models hyper-parameter tuning")
#         report = {}
#         exp_id_list = []
#         for model_key, model_value in models.items():
#             tuning_report, reports, best_model_so_far, exp_id_list_ = parameter_tuning(model_class=model_value,
#                                                                                        model_name=model_key,
#                                                                                        x_train=x_train,
#                                                                                        x_test=x_test,
#                                                                                        y_train=y_train,
#                                                                                        y_test=y_test,
#                                                                                        report_=report)
#             for i in exp_id_list_:
#                 exp_id_list.append(i)
#             report[model_key] = reports[model_key]
#             best_model_so_far_ = best_model_so_far
#             print(f"Model: {model_key}\nReport:\n{tuning_report}\n")
#             print("Experiment_ IDs: ", exp_id_list)

#             print(f"\nBest model so far: {best_model_so_far_[0]}\n")

#         cost = model_trainer(x_train=x_train,
#                              y_train=y_train,
#                              x_test=x_test,
#                              y_test=y_test,
#                              models=models,
#                              best_model_details=best_model_so_far_)

#         print(f"\nFinal Cost before Stacking and Voting Classifiers: {cost}\n")

#         best_model_sofar, best_models_with_params, best_estimators = best_model_finder(report=report, models=models)

#         sc_report, exp_id_stacking_clf = stacking_clf_trainer(best_estimators=best_estimators,
#                                                               models=models,
#                                                               best_model_so_far_=best_model_so_far_,
#                                                               x_train=x_train,
#                                                               y_train=y_train,
#                                                               x_test=x_test,
#                                                               y_test=y_test,
#                                                               report=report)
#         report['Stacked_Classifier'] = sc_report['Stacked_Classifier']
#         models['Stacked_Classifier'] = StackingClassifier
#         exp_id_list.append(exp_id_stacking_clf[0])

#         vc_report, exp_id_voting_clf = voting_clf_trainer(best_estimators=best_estimators,
#                                                           x_train=x_train,
#                                                           y_train=y_train,
#                                                           x_test=x_test,
#                                                           y_test=y_test,
#                                                           report=report)
#         report['Voting_Classifier'] = vc_report['Voting_Classifier']
#         models['Voting_Classifier'] = VotingClassifier
#         exp_id_list.append(exp_id_voting_clf)

#         mlflow_logger(exp_id=exp_id_list,
#                       should_register_champion_model=True,
#                       artifact_path=None)

#         best_model_sofar, best_models_with_params, best_estimators = best_model_finder(report=report, models=models)

#         print(f"\nBest model so far: {best_model_sofar[0]}\n")
#         if os.path.exists(self.model_config.hp_model_path):
#             pass
#         else:
#             os.makedirs(self.model_config.hp_model_path, exist_ok=True)
#         source = mlflow.search_registered_models(filter_string="tags.model_type ilike 'champion'")[0].latest_versions[0].source
#         pyfunc_model = mlflow.pyfunc.load_model(model_uri=source,
#                                                 dst_path=self.model_config.hp_model_path)

#         model = mlflow.pyfunc.load_model(f'{self.model_config.hp_model_path}/{pyfunc_model.metadata.artifact_path}')

#         artifact_path_dict = {'artifact_path_name': pyfunc_model.metadata.artifact_path}

#         save_yaml(file=artifact_path_dict, filepath=f"{self.model_config.root_dir}/artifact_path.yaml")
#         # save_yaml(file = model.metadata.artifact_path, filepath = self.model_config.root_dir)

#         y_pred = model.predict(data=x_test)

#         print(f"\nFinal metrics after Stacking and Voting Classifiers: {eval_metrics(y_test, y_pred)}\n")

#         best_model_sofar, best_models_with_params, best_estimators = best_model_finder(report=report, models=models)

#         print(f"\n\nBest Model Found: {best_model_sofar[0]}\n\n")
#         # print (f"Best models with params: {best_models_with_params}")
#         print(f"Best estimators: {[best_estimators[i][0] for i in range(len(best_estimators))]}\n\n")
#         print(f"Models checked so far: \n{list(models.keys())}\n\n")

#         logger.info(f"Saving metrics.yaml file at {self.metrics_config.metrics}")
#         save_yaml(file=report, filepath=self.metrics_config.metrics)

#         logger.info(f"Saving best_metrics.yaml file at {self.metrics_config.best_metric}")
#         save_yaml(file={best_model_sofar[0][0]: report[best_model_sofar[0][0]]}, filepath=self.metrics_config.best_metric)

#         logger.info(f"Saving model at {self.model_config.model_path}")
#         save_binary(file=models[best_model_sofar[0][0]], filepath=self.model_config.model_path)


# conf_obj = ConfigurationManager()
# stage_2_obj = conf_obj.get_stage2_processing_config()
# model_metrics_obj = conf_obj.get_metric_config()
# model_config_obj = conf_obj.get_model_config()
# data_split_obj = conf_obj.get_data_split_config()
# preprocessor_obj = conf_obj.get_preprocessor_config()
# stage_1_obj = conf_obj.get_stage1_processing_config()

# obj = model_tuning_tracking_component(stage_2_conf=stage_2_obj,
#                                       metrics_conf=model_metrics_obj,
#                                       model_conf=model_config_obj,
#                                       preprocessor_conf=preprocessor_obj,
#                                       data_split_conf=data_split_obj,
#                                       stage1_processor_conf=stage_1_obj)
# obj.models_tuning()

# # exp_id_voting_clf = mlflow.create_experiment(name = f"54_Voting_Classifier_54",
# #                                              tags = {"metrics": "['Balanced_Accuracy_Score', 'F1_Score', 'Accuracy_Score', 'Cost']"})
# # with mlflow.start_run(experiment_id = exp_id_voting_clf,
# #                       run_name = f"Voting_Classifier",
# #                       tags = {"run_type": "parent"}) as voting_clf_run:
# #     voting_clf_run_id = voting_clf_run.info.run_id
