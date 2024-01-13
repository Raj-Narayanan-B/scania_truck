from src.config.configuration_manager import ConfigurationManager
from src.entity.entity_config import DataSplitConf, Stage1ProcessingConf
from src.utils import train_test_splitter
import pandas as pd


class data_splitting_component:
    def __init__(self,
                 data_split_conf: DataSplitConf,
                 stage1_processor_conf: Stage1ProcessingConf) -> None:
        self.split_config = data_split_conf
        self.stage1_processor_config = stage1_processor_conf

    def data_splitting(self):
        df = pd.read_csv(self.stage1_processor_config.train_data_path)
        
        train_data_training_set,train_data_testing_set = train_test_splitter(df)
        train_data_training_set.to_csv(self.split_config.train_path,index=False)
        train_data_testing_set.to_csv(self.split_config.test_path,index=False)

obj = ConfigurationManager()
stage_1_obj = obj.get_stage1_processing_config()
splitter_obj = obj.get_data_split_config()

data_splitter_obj = data_splitting_component(data_split_conf = splitter_obj,
                                             stage1_processor_conf = stage_1_obj)
data_splitter_obj.data_splitting()