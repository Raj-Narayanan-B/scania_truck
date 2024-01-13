import pandas as pd
from src.utils import load_binary, hyper_parameter_tuning
from src.config.configuration_manager import ConfigurationManager
from src.entity.entity_config import ModelTrainerConf, Stage2ProcessingConf

class model_tuning_tracking_component:
    def __init__(self, model_conf: ModelTrainerConf,
                 stage2_conf: Stage2ProcessingConf) -> None:
        self.model_config = model_conf
        self.stage2_config = stage2_conf

    def model_tuning (self):
        model = load_binary(filepath = self.model_config.model_path)
        train_df = pd.read_csv(self.stage2_config.train_data_path).iloc[:1000,:]
        test_df = pd.read_csv(self.stage2_config.test_data_path).iloc[:1000,:]

        best_metric = hyper_parameter_tuning(model = model, train_data = train_df, test_data = test_df)
        
        print (best_metric)
