import pandas as pd
import os

from src.utils import load_yaml,save_yaml,save_binary,eval_metrics, parameter_tuning, best_model_finder
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
# from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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
        # model = load_binary(filepath = self.model_config.model_path)
        schema = load_yaml(SCHEMA_PATH)
        target = list(schema.Target.keys())[0]
        size = 10000
        logger.info("loading training and testing datasets")
        
        # if os.path.exists(self.stage_2_config.train_data_path) & os.path.exists(self.stage_2_config.test_data_path):
        #     train_df_shape = pd.read_csv(self.stage_2_config.train_data_path).shape
        #     test_df_shape = pd.read_csv(self.stage_2_config.test_data_path).shape
        #     if train_df_shape == size & test_df_shape == size
        if os.path.exists(self.preprocessor_config.preprocessor_path):
            os.remove(self.preprocessor_config.preprocessor_path)

        stage_3_data_split_obj = data_splitting_component(data_split_conf = self.split_config,
                                                          stage1_processor_conf = self.stage1_processor_config)
        pre_train_df, pre_test_df = stage_3_data_split_obj.data_splitting(size)

        stage_4_final_processing_obj = stage_4_final_processing_component(data_split_conf = self.split_config,
                                                                          stage_2_processor_conf = self.stage_2_config)
        train_df, test_df = stage_4_final_processing_obj.final_processing(pre_train_df, pre_test_df)
        
        # train_df = pd.read_csv(self.stage_2_config.train_data_path)
        # test_df = pd.read_csv(self.stage_2_config.test_data_path)

        print ("Train data's shape: ", train_df.shape)
        print ("Test data's shape: ", test_df.shape)

        print ("\nTrain data's target value_counts: ", train_df[target].value_counts())
        print ("\Test data's target value_counts: ", test_df[target].value_counts(),'\n')

        x_train = train_df.drop(columns = target)
        y_train = train_df[target]

        x_test = test_df.drop(columns = target)
        y_test = test_df[target]

        models = {'Logistic_Regression': LogisticRegression(solver = 'saga', max_iter=100), 
                  'SGD_Classifier': SGDClassifier(),
                  'Random Forest': RandomForestClassifier(), 
                  'Ada_Boost': AdaBoostClassifier(), 
                  'Grad_Boost': GradientBoostingClassifier(), 
                  'Bagging_Classifier': BaggingClassifier(), 
                  'ExtraTreesClassifier': ExtraTreesClassifier(), 
                  'Hist_Grad_Boost_Classifier': HistGradientBoostingClassifier(), 
                  'Decision_Tree_Classifier': DecisionTreeClassifier(),
                  'XGB_Classifier': XGBClassifier(),
                  'KNN_Classifier': KNeighborsClassifier(),
                  'MLP_Classifier': MLPClassifier()
                  }
        logger.info("Commencing models hyper-parameter tuning")
        report = {}
        for model_key, model_value in models.items():
            tuning_report,reports, best_model_so_far = parameter_tuning(model_class = model_value,
                                                                         model_name = model_key,
                                                                         x_train = x_train,
                                                                         x_test = x_test,
                                                                         y_train = y_train,
                                                                         y_test = y_test,
                                                                         report_ = report)
            report[model_key] = reports[model_key]
            best_model_so_far_ = best_model_so_far
            # costs = [value['Best_Cost'] for value in report.values()]
            # min_cost = min(costs)
            # best_model_so_far_ = [(i, min_cost, report[i]['Best_Params']) for i in report.keys() if min_cost == report[i]['Best_Cost']]

            print(f"\nBest model so far: {best_model_so_far_[0]}\n")

        model_class = models[best_model_so_far_[0][0]]
        print("Model: ",model_class)
        model = model_class.set_params(**best_model_so_far_[0][2])
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        cost = eval_metrics(y_true = y_test, y_pred = y_pred)
        print(f"Final Cost before Stacking and Voting Classifiers: {cost}")

        best_model_sofar,best_models_with_params, best_estimators = best_model_finder(report = report, models = models)
        
        stacked_classifier = StackingClassifier(estimators = best_estimators,
                                        final_estimator =  models[best_model_so_far_[0][0]].set_params(**best_model_so_far_[0][2]),
                                        cv = 5,
                                        n_jobs = -1,
                                        passthrough = False,
                                        verbose = 3)
        _, stacked_clf_report, best_model_so_far = parameter_tuning(model_class = stacked_classifier,
                                                                    model_name = 'Stacked_Classifier',
                                                                    x_train = x_train,
                                                                    x_test = x_test,
                                                                    y_train = y_train,
                                                                    y_test = y_test,
                                                                    report_ = report)
        
        report['Stacked_Classifier'] = stacked_clf_report['Stacked_Classifier']
        models['Stacked_Classifier'] = StackingClassifier

        voting_classifier_ = VotingClassifier(estimators = best_estimators,
                                                    voting = "hard",
                                                    weights = None,
                                                    n_jobs = -1,
                                                    verbose = True)
        voting_classifier_.fit(x_train,y_train)
        y_pred = voting_classifier_.predict(x_test)
        cost = eval_metrics(y_true = y_test, y_pred = y_pred)
        print(f"Voting Classifier Cost: {cost} \n")
        report['Voting_Classifier'] = {}
        report['Voting_Classifier']['Best_Params'] = voting_classifier_.get_params()
        report['Voting_Classifier']['Best_Cost'] = cost
        models['Voting_Classifier'] = VotingClassifier

        best_model_sofar, best_models_with_params, best_estimators = best_model_finder(report = report, models = models)

        print (f"Best Model Found: {best_model_sofar[0]}")
        print (f"Best models with params: {best_models_with_params}")
        print (f"Best estimators: {best_estimators}")

        logger.info(f"Saving metrics.yaml file at {self.metrics_config.metrics}")
        save_yaml(file = report, filepath = self.metrics_config.metrics)

        logger.info(f"Saving best_metrics.yaml file at {self.metrics_config.best_metric}")
        save_yaml(file = {best_model_sofar[0][0]: report[best_model_sofar[0][0]]}, filepath = self.metrics_config.best_metric)

        logger.info(f"Saving model at {self.model_config.model_path}")
        save_binary(file = models[best_model_sofar[0][0]],filepath = self.model_config.model_path)

        best_metric = parameter_tuning(model = model, train_data = train_df, test_data = test_df)
        
        print (best_metric)

conf_obj = ConfigurationManager()
stage_2_obj = conf_obj.get_stage2_processing_config()
model_metrics_obj = conf_obj.get_metric_config()
model_config_obj = conf_obj.get_model_config()
data_split_obj = conf_obj.get_data_split_config()
preprocessor_obj = conf_obj.get_preprocessor_config()
stage_1_obj = conf_obj.get_stage1_processing_config()

obj = model_tuning_tracking_component(stage_2_conf = stage_2_obj,
                                      metrics_conf = model_metrics_obj,
                                      model_conf = model_config_obj,
                                      preprocessor_conf = preprocessor_obj,
                                      data_split_conf = data_split_obj,
                                      stage1_processor_conf = stage_1_obj)
obj.models_tuning()