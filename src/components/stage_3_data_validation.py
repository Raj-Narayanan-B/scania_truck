from src.components.stage_2_initial_preprocessing import stage_2_initial_processing_component
from src.utils import data_validation
import pandas as pd


class data_validation_component(stage_2_initial_processing_component):
    def __init__(self):
        super().__init__()
        self.stage1_data = self.get_stage1_processing_config()
        self.data_validation = self.get_data_validation_config()

    def data_validation_(self, dataframe_: pd.DataFrame = None, cols_to_remove_: list = None, columns_with_0_std_dev_: list = None):

        if dataframe_ is None:
            dataframe_ = pd.read_csv(self.stage1_data.train_data_path)
            validated_data = data_validation(dataframe_)
            validated_data.to_csv(self.data_validation.validated_data, index=False)

            self.git_dvc_track([self.data_validation.validated_data])

        else:
            validated_data = data_validation(dataframe=dataframe_,
                                             cols_to_remove=cols_to_remove_,
                                             columns_with_0_std_dev=columns_with_0_std_dev_)
            return validated_data

# validation_obj = data_validation_component()

# validation_obj.data_validation_()
