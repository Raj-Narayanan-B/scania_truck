import pandas as pd
import os
# from typing import Any #type: ignore
from src.components.stage_3_data_split import data_splitting_component
# from src.entity.entity_config import DataSplitConf, Stage2ProcessingConf, PreprocessorConf
from src.utils import stage_2_processing_function


class stage_4_final_processing_component(data_splitting_component):
    def __init__(self):
        super().__init__()
        self.data_split_config = self.get_data_split_config()
        self.stage_2_processor_config = self.get_stage2_processing_config()
        self.preprocessor_config = self.get_preprocessor_config()

    def final_processing(self, *args):
        if os.path.exists(self.preprocessor_config.preprocessor_path):
            os.remove(self.preprocessor_config.preprocessor_path)

        if args:
            pre_processed_train_df = args[0]
            pre_processed_test_df = args[1]
            transformed_train_df = stage_2_processing_function(pre_processed_train_df)
            transformed_test_df = stage_2_processing_function(pre_processed_test_df)
            return (transformed_train_df, transformed_test_df)
        else:
            pre_processed_train_df = pd.read_csv(self.data_split_config.train_path)
            pre_processed_test_df = pd.read_csv(self.data_split_config.test_path)

            transformed_train_df = stage_2_processing_function(pre_processed_train_df)
            transformed_train_df.to_csv(self.stage_2_processor_config.train_data_path,
                                        index=False)

            transformed_test_df = stage_2_processing_function(pre_processed_test_df)
            transformed_test_df.to_csv(self.stage_2_processor_config.test_data_path,
                                       index=False)
            return (transformed_train_df, transformed_test_df)


# config = ConfigurationManager()
# data_split_obj = config.get_data_split_config()
# stage_2_config = config.get_stage2_processing_config()
# preprocessor_config = config.get_preprocessor_config()
# obj = stage_4_final_processing_component()
# obj.final_processing()
