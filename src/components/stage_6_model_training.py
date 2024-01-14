import pandas as pd
import json

# from sklearn.metrics import confusion_matrix

from src.config.configuration_manager import ConfigurationManager
from src.entity.entity_config import Stage2ProcessingConf, ModelMetricsConf, ModelTrainerConf
from src.constants import *
from src.utils import load_yaml,save_yaml,save_binary,eval_metrics, parameter_tuning, best_model_finder
from src import logger

class model_trainer_component:
    def __init__(self,
                 stage_2_conf: Stage2ProcessingConf,
                 metrics_conf: ModelMetricsConf,
                 model_conf: ModelTrainerConf) -> None:
        self.stage_2_config = stage_2_conf
        self.metrics_config = metrics_conf
        self.model_config = model_conf

    def model_training(self):
        schema = load_yaml(SCHEMA_PATH)
        target = list(schema.Target.keys())[0]
        logger.info("loading training and testing datasets")
        
        
        # metrics_list = ['balanced_accuracy_score','f1_score','accuracy_score']
        # logger.info("Commencing models training")
        # for model_name,model in models.items():

            
        #     print(model_name)
        #     model.fit(x_train,y_train)
        #     y_pred = model.predict(x_test)
            
        #     metrics = eval_metrics(y_true = y_test, y_pred = y_pred)

        #     print(metrics,'\n')

        #     report[model_name] = metrics

        # best_model_name_by_cost = min(report.keys(), key = lambda k: report[k].get('cost', 0)) 

        # print("\nBest Model before stacking and voting classifiers:")
        # print(models[best_model_name_by_cost])
        # print(report[best_model_name_by_cost],'\n')

        # best_models = sorted(report.items(), key = lambda x: x[1]['cost'])[:5]
        # best_models_names = [best_models[i][0] for i in range(len(best_models))]
        # best_estimators = {}
        # for i in best_models_names:
        #     best_estimators[i] = models[i]
        # best_estimators_copy = best_estimators.copy()
        # best_estimators_copy = list(zip(best_estimators_copy.keys(),best_estimators_copy.values()))

        # stacked_classifier = StackingClassifier(estimators = best_estimators_copy,
        #                                 final_estimator =  models[best_model_name_by_cost],
        #                                 cv = 5,
        #                                 n_jobs = -1,
        #                                 passthrough = False,
        #                                 verbose = 3)

        # stacked_classifier.fit(x_train,y_train)
        # y_pred = stacked_classifier.predict(x_test)
        # metrics = eval_metrics(y_true = y_test, y_pred = y_pred)
        # print(metrics,'\n')
        # report['Stacked_Classifier'] = metrics
        # models['Stacked_Classifier'] = stacked_classifier

        # voting_classifier_ = VotingClassifier(estimators = best_estimators_copy,
        #                                     voting = "hard",
        #                                     weights = None,
        #                                     n_jobs = -1,
        #                                     verbose = True)
        # voting_classifier_.fit(x_train,y_train)
        # y_pred = voting_classifier_.predict(x_test)
        # metrics = eval_metrics(y_true = y_test, y_pred = y_pred)
        # print(metrics,'\n')
        # report['Voting_Classifier'] = metrics
        # models['Voting_Classifier'] = voting_classifier_

        # best_model_name_by_cost = min(report.keys(), key = lambda k: report[k].get('cost', 0))
        # print("\nBest Model after stacking and voting classifiers:")
        # print(models[best_model_name_by_cost])
        # print(report[best_model_name_by_cost],'\n')

        # logger.info(f"Saving metrics.yaml file at {self.metrics_config.metrics}")
        # save_yaml(file = report, filepath = self.metrics_config.metrics)
        
        # logger.info(f"Saving best_metrics.yaml file at {self.metrics_config.best_metric}")
        # save_yaml(file = {best_model_name_by_cost: report[best_model_name_by_cost]}, filepath = self.metrics_config.best_metric)
                       
        # logger.info(f"Saving model at {self.model_config.model_path}")
        # save_binary(file = models[best_model_name_by_cost],filepath = self.model_config.model_path)
        


conf_obj = ConfigurationManager()
stage_2_obj = conf_obj.get_stage2_processing_config()
model_metrics_obj = conf_obj.get_metric_config()
model_config_obj = conf_obj.get_model_config()

obj = model_trainer_component(stage_2_conf = stage_2_obj,
                              metrics_conf = model_metrics_obj,
                              model_conf = model_config_obj)
obj.model_training()




