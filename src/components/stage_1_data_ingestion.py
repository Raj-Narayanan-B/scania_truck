from src.utils import DB_data_downloader
from src.config.configuration_manager import ConfigurationManager


class data_ingestion_component(ConfigurationManager):
    def __init__(self):
        super().__init__()
        self.astra_dB_data_config = self.get_astra_dB_data_config()
        self.data_config = self.get_data_path_config()

    def data_ingestion(self):
        train_data_config = [[self.astra_dB_data_config.train_data_1_secure_connect_bundle,
                              self.astra_dB_data_config.train_data_1_token,
                              self.astra_dB_data_config.train_data_1_key_space,
                              self.astra_dB_data_config.train_data_1_table,
                              self.data_config.train_data1],

                             [self.astra_dB_data_config.train_data_2_secure_connect_bundle,
                              self.astra_dB_data_config.train_data_2_token,
                              self.astra_dB_data_config.train_data_2_key_space,
                              self.astra_dB_data_config.train_data_2_table,
                              self.data_config.train_data2],

                             [self.astra_dB_data_config.train_data_3_secure_connect_bundle,
                              self.astra_dB_data_config.train_data_3_token,
                              self.astra_dB_data_config.train_data_3_key_space,
                              self.astra_dB_data_config.train_data_3_table,
                              self.data_config.train_data3]]

        test_data_config = [[self.astra_dB_data_config.test_data_1_secure_connect_bundle,
                             self.astra_dB_data_config.test_data_1_token,
                             self.astra_dB_data_config.test_data_1_key_space,
                             self.astra_dB_data_config.test_data_1_table,
                             self.data_config.test_data1],

                            [self.astra_dB_data_config.test_data_2_secure_connect_bundle,
                             self.astra_dB_data_config.test_data_2_token,
                             self.astra_dB_data_config.test_data_2_key_space,
                             self.astra_dB_data_config.test_data_2_table,
                             self.data_config.test_data2],

                            [self.astra_dB_data_config.test_data_3_secure_connect_bundle,
                             self.astra_dB_data_config.test_data_3_token,
                             self.astra_dB_data_config.test_data_3_key_space,
                             self.astra_dB_data_config.test_data_3_table,
                             self.data_config.test_data3]]

        for i in train_data_config:
            DB_data_downloader(i)

        for i in test_data_config:
            DB_data_downloader(i)

        self.git_dvc_track([self.data_config.train_data1, self.data_config.train_data2, self.data_config.train_data3,
                            self.data_config.test_data1, self.data_config.test_data2, self.data_config.test_data3])

# config_obj = ConfigurationManager()
# config_obj_ = config_obj.get_data_ingestion_config()
# obj = data_ingestion_component()
# obj.data_ingestion()
