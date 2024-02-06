import pandas as pd
# exp_count = 276
from src.utils import (parameter_tuning_2, model_trainer, best_model_finder,
                       eval_metrics, save_yaml, voting_clf_trainer, stacking_clf_trainer)

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
                              HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

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

    def models_tuning(self):
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
        test_df = pd.read_csv(self.stage_2_config.test_data_path)  # CHANGED

        # Initialize x_train, x_test, y_train, y_test
        x_train, y_train = train_df.drop(columns='class'), train_df['class']
        x_test, y_test = test_df.drop(columns='class'), test_df['class']

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
                  'Light_GBM': LGBMClassifier,
                  'KNN_Classifier': KNeighborsClassifier,
                  }
        logger.info("Commencing models' Hyper_Parameter Tuning")
        optuna_study_df, exp_id_list, model_accuracies, best_exp_id = parameter_tuning_2(models=models,  # CHANGED
                                                                                         client=client,
                                                                                         dataframe=dataframe)
        logger.info("Hyper_Parameter Tuning complete")
        # best_exp_id = '574'  #################CHANGED
        best_exp_ids = []
        best_exp_ids.append(best_exp_id)

        best_estimators_mlflow, final_estimator_mlflow, final_estimator = best_model_finder(models=models, client=client)
        print(f"\nBest_Estimators: {best_estimators_mlflow}\n")
        print(f"\nFinal_estimator: {final_estimator_mlflow}\n")

        metrics_final_estimator, y_pred_final_estimator, final_estimator_run_id = model_trainer(x_train=x_train, y_train=y_train,
                                                                                                x_test=x_test, y_test=y_test,
                                                                                                model=final_estimator,
                                                                                                model_dict=final_estimator_mlflow,
                                                                                                mlflow_experiment_id=best_exp_id,
                                                                                                client=client)

        metrics_stacking_clf, exp_id_stack_clf, y_pred_stacking_clf, stacking_clf_run_id = stacking_clf_trainer(best_estimators=best_estimators_mlflow,
                                                                                                                final_estimator=final_estimator,
                                                                                                                x_train=x_train, y_train=y_train,
                                                                                                                x_test=x_test, y_test=y_test,
                                                                                                                client=client)
        best_exp_ids.append(exp_id_stack_clf)

        metrics_voting_clf, exp_id_voting_clf, y_pred_voting_clf, voting_clf_run_id = voting_clf_trainer(best_estimators=best_estimators_mlflow,
                                                                                                         x_train=x_train, y_train=y_train,
                                                                                                         x_test=x_test, y_test=y_test,
                                                                                                         client=client)
        best_exp_ids.append(exp_id_voting_clf)

        # best_accuracy = max([metrics_voting_clf['Accuracy_Score'], metrics_stacking_clf['Accuracy_Score'], metrics_final_estimator['Accuracy_Score']])

        print(
            f"\nMetrics are: \n***************************\nStacking_CLF:\n{metrics_stacking_clf['Accuracy_Score']}\n**************************\nVoting_CLF:\n\
                {metrics_voting_clf['Accuracy_Score']}\n**************************\n{metrics_final_estimator['model'].__class__.__name__}:\n{metrics_final_estimator['Accuracy_Score']}\n\
                    **************************\n")

        report = {}
        report['Stacking_Classifier'] = eval_metrics(y_true=y_test, y_pred=y_pred_stacking_clf)
        report['Voting_Classifier'] = eval_metrics(y_true=y_test, y_pred=y_pred_voting_clf)
        report[f"{metrics_final_estimator['model'].__class__.__name__}"] = eval_metrics(y_true=y_test, y_pred=y_pred_final_estimator)

        logger.info(f"Saving metrics.yaml file at {self.metrics_config.metrics}")
        save_yaml(file=report, filepath=self.metrics_config.metrics)


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
