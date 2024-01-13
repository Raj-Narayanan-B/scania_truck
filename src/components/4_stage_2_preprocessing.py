import pandas as pd
from src.entity.entity_config import DataSplitConf,Stage2ProcessingConf
from src.config.configuration_manager import ConfigurationManager
from src.utils import save_binary, stage_2_processing_function


class stage_2_processing_component:
    def __init__(self, data_split_conf: DataSplitConf,
                 stage_2_processor_conf:Stage2ProcessingConf) -> None:
        self.data_split_config = data_split_conf
        self.stage_2_processor_config = stage_2_processor_conf

    def stage_2_processing(self):
        pre_processed_train_df = pd.read_csv(self.data_split_config.train_path)
        pre_processed_test_df = pd.read_csv(self.data_split_config.test_path)


        transformed_train_df = stage_2_processing_function(pre_processed_train_df)
        transformed_train_df.to_csv(self.stage_2_processor_config.train_data_path,
                                    index = False)
        
        transformed_test_df = stage_2_processing_function(pre_processed_test_df)
        transformed_test_df.to_csv(self.stage_2_processor_config.test_data_path,
                                   index = False)
        
config = ConfigurationManager()
data_split_obj = config.get_data_split_config()
stage_2_config = config.get_stage2_processing_config()
obj = stage_2_processing_component(data_split_conf = data_split_obj,
                                   stage_2_processor_conf = stage_2_config)
obj.stage_2_processing()