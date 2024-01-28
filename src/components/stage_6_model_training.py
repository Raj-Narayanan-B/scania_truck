import pandas as pd
import json
import mlflow.pyfunc

# from sklearn.metrics import confusion_matrix

from src.config.configuration_manager import ConfigurationManager
from src.entity.entity_config import (DataSplitConf, Stage2ProcessingConf, 
                                      ModelMetricsConf, ModelTrainerConf)
from src.constants import *
from src.utils import load_yaml,save_yaml,save_binary,eval_metrics, parameter_tuning, best_model_finder
from src import logger
from src.components.stage_3_data_split import data_splitting_component
from src.components.stage_4_final_preprocessing import stage_4_final_processing_component

class model_trainer_component:
    def __init__(self,
                 data_split_conf: DataSplitConf,
                 stage_2_conf: Stage2ProcessingConf,
                 metrics_conf: ModelMetricsConf,
                 model_conf: ModelTrainerConf) -> None:
        # self.stage_1_config = stage_1_conf
        self.data_split_config = data_split_conf
        self.stage_2_config = stage_2_conf
        self.metrics_config = metrics_conf
        self.model_config = model_conf

    def model_training(self):
        schema = load_yaml(SCHEMA_PATH)
        target = list(schema.Target.keys())[0]
        logger.info("loading training and testing datasets")

        data_split_class_obj = data_splitting_component(data_split_conf = data_split_obj, 
                                                        stage1_processor_conf = stage_1_obj)

        final_processing_class_obj = stage_4_final_processing_component(data_split_conf = data_split_obj,
                                                                        stage_2_processor_conf = stage_2_obj,
                                                                        preprocessor_conf = preprocessor_obj)
        
        data_split_class_obj.data_splitting()
        final_processing_class_obj.final_processing()

        train_df = pd.read_csv(self.stage_2_config.train_data_path)
        test_df = pd.read_csv(self.stage_2_config.test_data_path)

        x_train,y_train = train_df.drop(columns = target), train_df[target]
        x_test,y_test = test_df.drop(columns = target), test_df[target]

        print(f"\nx_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
        print(f"\nNA values in x_train: {x_train.isna().sum().unique()}")
        print(f"NA values in x_test: {x_test.isna().sum().unique()}")
        print(f"\nTarget value counts in y_train: {y_train.value_counts()}")
        print(f"\nTarget value counts in y_test: {y_test.value_counts()}")

        



obj = ConfigurationManager()
stage_1_obj = obj.get_stage1_processing_config()
data_split_obj = obj.get_data_split_config()
stage_2_obj = obj.get_stage2_processing_config()
model_metrics_obj = obj.get_metric_config()
model_config_obj = obj.get_model_config()
preprocessor_obj = obj.get_preprocessor_config()

model_trainer_obj = model_trainer_component(data_split_conf = data_split_obj,
                              stage_2_conf = stage_2_obj,
                              metrics_conf = model_metrics_obj,
                              model_conf = model_config_obj)



model_trainer_obj.model_training()




