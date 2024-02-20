from src.components.stage_1_data_ingestion import data_ingestion_component
from src.utils import stage_1_processing_function, schema_saver
import pandas as pd
from src import logger


class stage_2_initial_processing_component(data_ingestion_component):
    def __init__(self):
        super().__init__()
        self.preprocess_config = self.get_stage1_processing_config()
        self.data_config = self.get_data_path_config()
        # self.split_config = self.get_data_split_config()

    def initial_processing(self):
        train_df_1 = pd.read_csv(self.data_config.train_data1)
        train_df_2 = pd.read_csv(self.data_config.train_data2)
        train_df_3 = pd.read_csv(self.data_config.train_data3)
        train_df = stage_1_processing_function([train_df_1, train_df_2, train_df_3])

        test_df_1 = pd.read_csv(self.data_config.test_data1)
        test_df_2 = pd.read_csv(self.data_config.test_data2)
        test_df_3 = pd.read_csv(self.data_config.test_data3)
        test_df = stage_1_processing_function([test_df_1, test_df_2, test_df_3])

        train_df.to_csv(self.preprocess_config.train_data_path, index=False)
        test_df.to_csv(self.preprocess_config.test_data_path, index=False)

        logger.info("Saving schema for entire/original data")
        schema_saver(dataframe=train_df,
                     labels=['Features', 'Target'],
                     mode='w')

        self.git_dvc_track([self.preprocess_config.train_data_path,
                            self.preprocess_config.test_data_path])


# config_obj = ConfigurationManager()
# preprocessing_obj = config_obj.get_stage1_processing_config()
# data_obj = config_obj.get_data_path_config()
# split_obj = config_obj.get_data_split_config()
# obj = stage_2_initial_processing_component()
# obj.initial_processing()
