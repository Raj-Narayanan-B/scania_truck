# from src.constants import CONFIG_PATH, PARAMS_PATH, SCHEMA_PATH
# from src.entity.entity_config import DataIngestionConf
from src.utils import DB_data_loader
from src.config.configuration_manager import ConfigurationManager


class data_ingestion_component(ConfigurationManager):
    def __init__(self):
        super().__init__()
        self.ingestion_config = self.get_data_ingestion_config()

    def data_ingestion(self):
        train_data_config = [[self.ingestion_config.train_data_1_secure_connect_bundle, self.ingestion_config.train_data_1_token,
                              self.ingestion_config.train_data_1_key_space, self.ingestion_config.train_data_1_table, self.ingestion_config.train_data_1_path],
                             [self.ingestion_config.train_data_2_secure_connect_bundle, self.ingestion_config.train_data_2_token,
                              self.ingestion_config.train_data_2_key_space, self.ingestion_config.train_data_2_table, self.ingestion_config.train_data_2_path],
                             [self.ingestion_config.train_data_3_secure_connect_bundle, self.ingestion_config.train_data_3_token,
                              self.ingestion_config.train_data_3_key_space, self.ingestion_config.train_data_3_table, self.ingestion_config.train_data_3_path]]

        test_data_config = [[self.ingestion_config.test_data_1_secure_connect_bundle, self.ingestion_config.test_data_1_token,
                             self.ingestion_config.test_data_1_key_space, self.ingestion_config.test_data_1_table, self.ingestion_config.test_data_1_path],
                            [self.ingestion_config.test_data_2_secure_connect_bundle, self.ingestion_config.test_data_2_token,
                             self.ingestion_config.test_data_2_key_space, self.ingestion_config.test_data_2_table, self.ingestion_config.test_data_2_path],
                            [self.ingestion_config.test_data_3_secure_connect_bundle, self.ingestion_config.test_data_3_token,
                             self.ingestion_config.test_data_3_key_space, self.ingestion_config.test_data_3_table, self.ingestion_config.test_data_3_path]]

        for i in train_data_config:
            DB_data_loader(i)

        for i in test_data_config:
            DB_data_loader(i)


# config_obj = ConfigurationManager()
# config_obj_ = config_obj.get_data_ingestion_config()
# obj = data_ingestion_component(config_obj_)
# obj.data_ingestion()
