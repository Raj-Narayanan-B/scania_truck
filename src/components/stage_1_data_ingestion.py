from src.entity.entity_config import DataIngestionConf
from src.utils import DB_data_loader
from src.config.configuration_manager import ConfigurationManager


class data_ingestion_component:
    def __init__(self, config: DataIngestionConf):
        self.config = config

    def data_ingestion(self):
        train_data_config = [[self.config.train_data_1_secure_connect_bundle, self.config.train_data_1_token,
                              self.config.train_data_1_key_space, self.config.train_data_1_table, self.config.train_data_1_path],
                             [self.config.train_data_2_secure_connect_bundle, self.config.train_data_2_token,
                              self.config.train_data_2_key_space, self.config.train_data_2_table, self.config.train_data_2_path],
                             [self.config.train_data_3_secure_connect_bundle, self.config.train_data_3_token,
                              self.config.train_data_3_key_space, self.config.train_data_3_table, self.config.train_data_3_path]]

        test_data_config = [[self.config.test_data_1_secure_connect_bundle, self.config.test_data_1_token,
                             self.config.test_data_1_key_space, self.config.test_data_1_table, self.config.test_data_1_path],
                            [self.config.test_data_2_secure_connect_bundle, self.config.test_data_2_token,
                             self.config.test_data_2_key_space, self.config.test_data_2_table, self.config.test_data_2_path],
                            [self.config.test_data_3_secure_connect_bundle, self.config.test_data_3_token,
                             self.config.test_data_3_key_space, self.config.test_data_3_table, self.config.test_data_3_path]]

        for i in train_data_config:
            DB_data_loader(i)

        for i in test_data_config:
            DB_data_loader(i)


config_obj = ConfigurationManager()
config_obj_ = config_obj.get_data_ingestion_config()
obj = data_ingestion_component(config_obj_)
obj.data_ingestion()
