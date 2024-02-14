import os
import shutil  # type: ignore
from src.config.configuration_manager import ConfigurationManager
from src.constants import REPO, BUCKET, TEST_DATA, TRAIN_DATA
from src.utils import DB_data_uploader
from dagshub import get_repo_bucket_client


class data_db_uploader_component(ConfigurationManager):
    def __init__(self):
        super().__init__()
        self.astra_dB_data_config = self.get_astra_dB_data_config()
        self.data_config = self.get_data_path_config()

    def s3_data_download(self):
        filepath, file = os.path.split(self.data_config.temp_train_data)
        os.makedirs(filepath, exist_ok=True)

        s3 = get_repo_bucket_client(REPO + '/' + BUCKET)
        if not os.path.exists(self.data_config.temp_train_data):
            s3.download_file(
                Bucket=BUCKET,  # name of the repo
                Key=TRAIN_DATA,  # remote path from where to download the file
                Filename=self.data_config.temp_train_data,  # local path where to download the file
            )

        if not os.path.exists(self.data_config.temp_test_data):
            s3.download_file(
                Bucket=BUCKET,  # name of the repo
                Key=TEST_DATA,  # remote path from where to download the file
                Filename=self.data_config.temp_test_data,  # local path where to download the file
            )

    def data_db_upload(self):
        train_data_config = [[self.data_config.temp_train_data],

                             [self.astra_dB_data_config.train_data_1_secure_connect_bundle,
                              self.astra_dB_data_config.train_data_1_token,
                              self.astra_dB_data_config.train_data_1_key_space,
                              self.astra_dB_data_config.train_data_1_table,
                              self.data_config.temp_train_data1],

                             [self.astra_dB_data_config.train_data_2_secure_connect_bundle,
                              self.astra_dB_data_config.train_data_2_token,
                              self.astra_dB_data_config.train_data_2_key_space,
                              self.astra_dB_data_config.train_data_2_table,
                              self.data_config.temp_train_data2],

                             [self.astra_dB_data_config.train_data_3_secure_connect_bundle,
                              self.astra_dB_data_config.train_data_3_token,
                              self.astra_dB_data_config.train_data_3_key_space,
                              self.astra_dB_data_config.train_data_3_table,
                              self.data_config.temp_train_data3]]

        test_data_config = [[self.data_config.temp_test_data],

                            [self.astra_dB_data_config.test_data_1_secure_connect_bundle,
                             self.astra_dB_data_config.test_data_1_token,
                             self.astra_dB_data_config.test_data_1_key_space,
                             self.astra_dB_data_config.test_data_1_table,
                             self.data_config.temp_test_data1],

                            [self.astra_dB_data_config.test_data_2_secure_connect_bundle,
                             self.astra_dB_data_config.test_data_2_token,
                             self.astra_dB_data_config.test_data_2_key_space,
                             self.astra_dB_data_config.test_data_2_table,
                             self.data_config.temp_test_data2],

                            [self.astra_dB_data_config.test_data_3_secure_connect_bundle,
                             self.astra_dB_data_config.test_data_3_token,
                             self.astra_dB_data_config.test_data_3_key_space,
                             self.astra_dB_data_config.test_data_3_table,
                             self.data_config.temp_test_data3]]

        for i in [train_data_config, test_data_config]:
            DB_data_uploader(i)

        shutil.rmtree(self.data_config.temp_dir_root)


data_db_obj = data_db_uploader_component()
data_db_obj.s3_data_download()
data_db_obj.data_db_upload()
