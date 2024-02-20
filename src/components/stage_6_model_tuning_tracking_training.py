import pandas as pd
from src.utils import (parameter_tuning_2, model_trainer, best_model_finder,
                       eval_metrics, save_yaml, voting_clf_trainer, stacking_clf_trainer)

from src.components.stage_5_final_preprocessing import stage_5_final_processing_component
from src import logger
import os
# from src.constants import MLFLOW_URI

from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,  # noqa
                              GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier  # noqa
from xgboost import XGBClassifier  # noqa
from lightgbm import LGBMClassifier  # noqa
from sklearn.neighbors import KNeighborsClassifier  # noqa

from mlflow.client import MlflowClient


class model_tuning_tracking_component(stage_5_final_processing_component):
    def __init__(self):
        super().__init__()
        self.stage_2_config = self.get_stage2_processing_config()
        self.metrics_config = self.get_metric_config()
        self.preprocessor_config = self.get_preprocessor_config()
        self.model_config = self.get_model_config()
        self.split_config = self.get_data_split_config()
        self.stage1_processor_config = self.get_stage1_processing_config()

    def models_tuning(self):
        logger.info("loading training and testing datasets")
        params = self.params_path
        train_df = pd.read_csv(self.stage_2_config.train_data_path)
        test_df = pd.read_csv(self.stage_2_config.test_data_path)  # CHANGED

        # Initialize x_train, x_test, y_train, y_test
        x_train, y_train = train_df.drop(columns='class'), train_df['class']
        x_test, y_test = test_df.drop(columns='class'), test_df['class']

        client = MlflowClient(tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),
                              registry_uri=os.getenv('MLFLOW_TRACKING_URI'))
        # client = MlflowClient()

        dataframe = pd.read_csv(self.stage1_processor_config.train_data_path)

        models = {key: eval(value) for key, value in params['models'].items()}

        logger.info("Commencing models' Hyper_Parameter Tuning")
        optuna_study_df, exp_id_list, model_accuracies, best_exp_id = parameter_tuning_2(models=models,  # CHANGED
                                                                                         client=client,
                                                                                         dataframe=dataframe,
                                                                                         optuna_trials_study_df_path=self.metrics_config.model_trial_study_df)
        logger.info("Hyper_Parameter Tuning complete")
        # best_exp_id = '574'  #################CHANGED
        best_exp_ids = []
        best_exp_ids.append(best_exp_id)

        best_estimators_mlflow, final_estimator_mlflow, final_estimator = best_model_finder(models=models, client=client)
        print(f"\nBest_Estimators: {best_estimators_mlflow}\n")
        print(f"\nFinal_estimator: {final_estimator_mlflow}\n")

        metrics_final_estimator, y_pred_final_estimator = model_trainer(x_train=x_train, y_train=y_train,
                                                                        x_test=x_test, y_test=y_test,
                                                                        model=final_estimator,
                                                                        model_dict=final_estimator_mlflow,
                                                                        mlflow_experiment_id=best_exp_id,
                                                                        client=client,
                                                                        model_path=self.model_config.final_estimator_path)

        metrics_stacking_clf, exp_id_stack_clf, y_pred_stacking_clf = stacking_clf_trainer(best_estimators=best_estimators_mlflow,
                                                                                           final_estimator=final_estimator,
                                                                                           x_train=x_train, y_train=y_train,
                                                                                           x_test=x_test, y_test=y_test,
                                                                                           client=client,
                                                                                           model_path=self.model_config.stacking_classifier_path)
        best_exp_ids.append(exp_id_stack_clf)

        metrics_voting_clf, exp_id_voting_clf, y_pred_voting_clf = voting_clf_trainer(best_estimators=best_estimators_mlflow,
                                                                                      x_train=x_train, y_train=y_train,
                                                                                      x_test=x_test, y_test=y_test,
                                                                                      client=client,
                                                                                      model_path=self.model_config.voting_classifier_path)
        best_exp_ids.append(exp_id_voting_clf)

        print(f"\nMetrics are: \n***************************\nStacking_CLF:\n{metrics_stacking_clf['Accuracy_Score']}\n**************************\nVoting_CLF:\n{metrics_voting_clf['Accuracy_Score']}\n**************************\n{metrics_final_estimator['model'].__class__.__name__}:\n{metrics_final_estimator['Accuracy_Score']}\n**************************\n")  # noqa

        report = {}
        report['Stacking_Classifier'] = eval_metrics(y_true=y_test, y_pred=y_pred_stacking_clf)
        report['Voting_Classifier'] = eval_metrics(y_true=y_test, y_pred=y_pred_voting_clf)
        report[f"{metrics_final_estimator['model'].__class__.__name__}"] = eval_metrics(y_true=y_test, y_pred=y_pred_final_estimator)

        logger.info(f"Saving metrics.yaml file at {self.metrics_config.metrics}")
        save_yaml(file=report, filepath=self.metrics_config.metrics)

        self.git_dvc_track([self.model_config.final_estimator_path,
                            self.model_config.stacking_classifier_path,
                            self.model_config.voting_classifier_path,
                            self.metrics_config.model_trial_study_df])


# conf_obj = ConfigurationManager()
# stage_2_obj = conf_obj.get_stage2_processing_config()
# model_metrics_obj = conf_obj.get_metric_config()
# model_config_obj = conf_obj.get_model_config()
# data_split_obj = conf_obj.get_data_split_config()
# preprocessor_obj = conf_obj.get_preprocessor_config()
# stage_1_obj = conf_obj.get_stage1_processing_config()

# obj = model_tuning_tracking_component()
# obj.models_tuning()
