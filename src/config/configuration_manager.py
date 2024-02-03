from src.constants import *
from src.entity.entity_config import (DataIngestionConf,
                                      DataPathConf,
                                      Stage1ProcessingConf,
                                      Stage2ProcessingConf,
                                      PreprocessorConf,
                                      DataSplitConf,
                                      ModelTrainerConf,
                                      ModelMetricsConf)
from src.utils import load_yaml

class ConfigurationManager:
    def __init__(self,
                 config = CONFIG_PATH,  
                 params = PARAMS_PATH,
                 schema = SCHEMA_PATH
                 ):
        self.config = config
        self.config_path = load_yaml(config)
        self.params = params
        self.params_path = load_yaml(params)
        self.schema = schema
        self.schema_path = load_yaml(schema)

    def get_data_ingestion_config(self) -> DataIngestionConf:
        config = self.config_path.data_ingestion_config
        data_ingestion = DataIngestionConf(
            train_data_1_secure_connect_bundle = config.train_data1,
            train_data_1_token = config.train_data1.token,
            train_data_1_key_space = config.train_data1.key_space_train_data_1,
            train_data_1_table = config.train_data1.table_train_data_1,
            train_data_1_path = config.train_data1.path,

            train_data_2_secure_connect_bundle = config.train_data2,
            train_data_2_token = config.train_data2.token,
            train_data_2_key_space = config.train_data2.key_space_train_data_2,
            train_data_2_table = config.train_data2.table_train_data_2,
            train_data_2_path = config.train_data2.path,

            train_data_3_secure_connect_bundle = config.train_data3,
            train_data_3_token = config.train_data3.token,
            train_data_3_key_space = config.train_data3.key_space_train_data_3,
            train_data_3_table = config.train_data3.table_train_data_3,
            train_data_3_path = config.train_data3.path,

            test_data_1_secure_connect_bundle = config.test_data1,
            test_data_1_token = config.test_data1.token,
            test_data_1_key_space = config.test_data1.key_space_test_data_1,
            test_data_1_table = config.test_data1.table_test_data_1,
            test_data_1_path = config.test_data1.path,

            test_data_2_secure_connect_bundle = config.test_data2,
            test_data_2_token = config.test_data2.token,
            test_data_2_key_space = config.test_data2.key_space_test_data_2,
            test_data_2_table = config.test_data2.table_test_data_2,
            test_data_2_path = config.test_data2.path,

            test_data_3_secure_connect_bundle = config.test_data3,
            test_data_3_token = config.test_data3.token,
            test_data_3_key_space = config.test_data3.key_space_test_data_3,
            test_data_3_table = config.test_data3.table_test_data_3,
            test_data_3_path = config.test_data3.path,

            root_directory= config.root_dir
            )
        return data_ingestion
    
    def get_data_path_config(self) -> DataPathConf:
        config = self.config_path.data_path_config
        data_path_config = DataPathConf(
            train_data1 = config.train_data1,
            train_data2 = config.train_data2,
            train_data3 = config.train_data3,
            test_data1 = config.test_data1,
            test_data2 = config.test_data2,
            test_data3 = config.test_data3
        )
        return (data_path_config)

    def get_stage1_processing_config(self) -> Stage1ProcessingConf:
        config = self.config_path.data_stage_1_processing_config
        stage1_processing_config = Stage1ProcessingConf(
            root_dir = config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path
        )
        return stage1_processing_config
    
    def get_stage2_processing_config(self) -> Stage2ProcessingConf:
        config = self.config_path.data_stage_2_processing_config
        stage2_processing_config = Stage2ProcessingConf(
            root_dir = config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path
        )
        return stage2_processing_config
    
    def get_preprocessor_config(self) -> PreprocessorConf:
        config = self.config_path.preprocessor
        preprocessor_config = PreprocessorConf(
            root_dir = config.root_dir,
            preprocessor_path = config.preprocessor_path
        )
        return preprocessor_config

    def get_data_split_config(self) -> DataSplitConf:
        config = self.config_path.data_split_config
        data_split_config = DataSplitConf(
            root_dir = config.root_dir,
            train_path = config.train_batch_path,
            test_path = config.test_batch_path
        )
        return data_split_config
    
    def get_model_config(self) -> ModelTrainerConf:
        config = self.config_path.model_trainer
        model_config = ModelTrainerConf(
            root_dir = config.root_dir,
            model_path = config.model_path,
            hp_model_path = config.hp_model_path_
        )
        return model_config

    def get_metric_config(self) ->ModelMetricsConf:
        config = self.config_path.model_metrics
        metrics_config = ModelMetricsConf(
            root_dir = config.root_dir,
            metrics = config.metrics,
            best_metric = config.best_metric
        )
        return metrics_config


# obj = ConfigurationManager()
# print(obj.get_data_ingestion_config().train_data_2_token)