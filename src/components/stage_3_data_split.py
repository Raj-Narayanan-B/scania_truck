from src.entity.entity_config import DataSplitConf, Stage1ProcessingConf
from src.utils import train_test_splitter
import pandas as pd


class data_splitting_component:
    def __init__(self,
                 data_split_conf: DataSplitConf,
                 stage1_processor_conf: Stage1ProcessingConf) -> None:
        self.split_config = data_split_conf
        self.stage1_processor_config = stage1_processor_conf

    def data_splitting(self, *args):
        if args:
            self.size = args[0]
            print("Size: ", self.size)
            df = pd.read_csv(self.stage1_processor_config.train_data_path).iloc[:self.size, :]
            train_data_training_set, train_data_testing_set = train_test_splitter(df)
            print("Pre_train_data shape: ", train_data_training_set.shape,
                  "\nPre_test_data shape: ", train_data_testing_set.shape)
            return (train_data_training_set, train_data_testing_set)
        else:
            self.size = None
            print("Size: Full")
            df = pd.read_csv(self.stage1_processor_config.train_data_path).iloc[:self.size, :]
            train_data_training_set, train_data_testing_set = train_test_splitter(df)
            print("Pre_train_data shape: ", train_data_training_set.shape,
                  "\nPre_test_data shape: ", train_data_testing_set.shape)

            train_data_training_set.to_csv(self.split_config.train_path, index=False)
            train_data_testing_set.to_csv(self.split_config.test_path, index=False)

            return (train_data_training_set, train_data_testing_set)

# obj = ConfigurationManager()
# stage_1_obj = obj.get_stage1_processing_config()
# splitter_obj = obj.get_data_split_config()

# data_splitter_obj = data_splitting_component(data_split_conf = splitter_obj,
#                                              stage1_processor_conf = stage_1_obj)
# data_splitter_obj.data_splitting()
