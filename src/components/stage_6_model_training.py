import pandas as pd
import json
import mlflow.pyfunc

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
        


conf_obj = ConfigurationManager()
stage_2_obj = conf_obj.get_stage2_processing_config()
model_metrics_obj = conf_obj.get_metric_config()
model_config_obj = conf_obj.get_model_config()

obj = model_trainer_component(stage_2_conf = stage_2_obj,
                              metrics_conf = model_metrics_obj,
                              model_conf = model_config_obj)
obj.model_training()




