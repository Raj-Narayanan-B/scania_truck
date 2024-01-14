from src.config.configuration_manager import ConfigurationManager
from src.entity.entity_config import Stage1ProcessingConf,DataPathConf,DataSplitConf
from src.utils import stage_1_processing_function,schema_saver,train_test_splitter
import pandas as pd
import os

class stage_2_initial_processing_component:
    def __init__(self, preprocess_conf: Stage1ProcessingConf, data_conf: DataPathConf, split_conf: DataSplitConf) -> None:
        self.preprocess_config = preprocess_conf
        self.data_config = data_conf
        self.split_config = split_conf

    def initial_processing(self):
        train_df_1 = pd.read_csv(self.data_config.train_data1)
        train_df_2 = pd.read_csv(self.data_config.train_data2)
        train_df_3 = pd.read_csv(self.data_config.train_data3)
        train_df = stage_1_processing_function([train_df_1,train_df_2,train_df_3])

        test_df_1 = pd.read_csv(self.data_config.test_data1)
        test_df_2 = pd.read_csv(self.data_config.test_data2)
        test_df_3 = pd.read_csv(self.data_config.test_data3)
        test_df = stage_1_processing_function([test_df_1,test_df_2,test_df_3])

        train_df.to_csv(self.preprocess_config.train_data_path,index=False)
        test_df.to_csv(self.preprocess_config.test_data_path,index=False)

        # To create the schema.yaml
        schema_saver(train_df)

config_obj = ConfigurationManager()
preprocessing_obj = config_obj.get_stage1_processing_config()
data_obj = config_obj.get_data_path_config()
split_obj = config_obj.get_data_split_config()
obj = stage_2_initial_processing_component(preprocess_conf = preprocessing_obj,
                                   data_conf = data_obj,
                                   split_conf=split_obj)
obj.initial_processing()