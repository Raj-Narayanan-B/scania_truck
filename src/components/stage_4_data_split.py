from src.components.stage_3_data_validation import data_validation_component
from src.utils import train_test_splitter
import pandas as pd


class data_splitting_component(data_validation_component):
    def __init__(self):
        super().__init__()
        self.split_config = self.get_data_split_config()
        self.data_validation = self.get_data_validation_config()

    def data_splitting(self, *args):
        if args:
            self.size = args[0]
            print("Size: ", self.size)
            df = pd.read_csv(self.data_validation.validated_data).iloc[:self.size, :]
            train_data_training_set, train_data_testing_set = train_test_splitter(df)
            print("Pre_train_data shape: ", train_data_training_set.shape,
                  "\nPre_test_data shape: ", train_data_testing_set.shape)
            return (train_data_training_set, train_data_testing_set)
        else:
            self.size = None
            print("Size: Full")
            df = pd.read_csv(self.data_validation.validated_data).iloc[:self.size, :]
            train_data_training_set, train_data_testing_set = train_test_splitter(df)
            print("Pre_train_data shape: ", train_data_training_set.shape,
                  "\nPre_test_data shape: ", train_data_testing_set.shape)

            train_data_training_set.to_csv(self.split_config.train_path, index=False)
            train_data_testing_set.to_csv(self.split_config.test_path, index=False)

            self.git_dvc_track([self.split_config.train_path,
                                self.split_config.test_path])

            return (train_data_training_set, train_data_testing_set)

# obj = ConfigurationManager()
# stage_1_obj = obj.get_stage1_processing_config()
# splitter_obj = obj.get_data_split_config()

# data_splitter_obj = data_splitting_component(data_split_conf = splitter_obj,
#                                              stage1_processor_conf = stage_1_obj)
# data_splitter_obj.data_splitting()
