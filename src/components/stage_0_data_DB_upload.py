import os
import shutil  # type: ignore
from src import logger
from src.config.configuration_manager import ConfigurationManager
from src.constants import REPO, BUCKET, TEST_DATA, TRAIN_DATA
from src.utils import (DB_data_uploader, load_yaml, save_yaml, file_lineage_s3_files_refresher,
                       file_lineage_adder, file_lineage_updater, file_lineage_reverse_updater)
from dagshub import get_repo_bucket_client
from ensure import ensure_annotations


class s3_data_downloader(ConfigurationManager):
    def __init__(self) -> None:
        super().__init__()
        self.s3 = get_repo_bucket_client(REPO + '/' + BUCKET)

    def s3_data_download(self, key: str, filepath: str):
        """
            A function to download a file from the project's dagshub S3 bucket

            Parameters
            ----------

            key: str; default=None
                key refers to the filename in s3 bucket.
            filepath: str; default=None
                filepath refers to the local filepath where the downloaded file should be saved
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)
        self.s3.download_file(
            Bucket=BUCKET,
            Key=key,  # files_from_s3_dict['files_for_model_training'][file_name],
            Filename=filepath  # self.data_config.temp_dir_root + f"/{files_from_s3_dict['files_for_model_training'][file_name]}"
        )


class file_tracker_component(s3_data_downloader):
    def __init__(self):
        super().__init__()
        self.astra_dB_data_config = self.get_astra_dB_data_config()
        self.data_config = self.get_data_path_config()

    @ensure_annotations
    def file_lineage_tracker(self, update_list: list = None, reverse_update_list: list = None,
                             add_list: list = None):
        files_in_s3_ = self.s3.list_objects_v2(Bucket=BUCKET)['Contents']

        if os.path.exists(self.data_config.data_from_s3):
            old_files_ = load_yaml(self.data_config.data_from_s3)

            old_files = file_lineage_s3_files_refresher(files_in_s3=files_in_s3_, old_files=old_files_)

            old_files_for_model_training = old_files['files_for_model_training']
            old_files_predicted = old_files['files_predicted']
            old_files_to_predict = old_files['files_to_predict']

            if add_list:
                old_files_predicted = file_lineage_adder(add_list, old_files_predicted)

            if update_list:
                old_files_predicted, old_files_to_predict = file_lineage_updater(update_list=update_list,
                                                                                 old_files_predicted=old_files_predicted,
                                                                                 old_files_to_predict=old_files_to_predict)

            if reverse_update_list:
                old_files_predicted, old_files_to_predict = file_lineage_reverse_updater(reverse_update_list=reverse_update_list,
                                                                                         old_files_predicted=old_files_predicted,
                                                                                         old_files_to_predict=old_files_to_predict)

            files_from_s3_dict = {}
            files_from_s3_dict['files_for_model_training'] = old_files_for_model_training
            files_from_s3_dict['files_to_predict'] = old_files_to_predict
            files_from_s3_dict['files_predicted'] = old_files_predicted

            save_yaml(files_from_s3_dict,
                      filepath=self.data_config.data_from_s3,
                      mode='w')
            # predicted_files_counter = 1
            # files_to_predict_counter = 1

            # # get the list version of the old files
            # old_files_list = list(old_files['files_predicted'].values())
            # all_old_files_list = list(old_files['files_predicted'].values()) + list(old_files['files_to_predict'].values())

            # # initialize the values to be written into the file_lineage file
            # # the files are gotten from S3 - this will be the initial input values list
            # files_ = []
            # for i in range(len(files_in_s3)):
            #     files_.append(files_in_s3[i]['Key'])

            # # get the files that were manually added previously (manually - added via bulk/batch prediction option in html page)
            # # this will work only if "add_predicted_files" parameter is given.
            # previously_added_files = list(set(all_old_files_list) - set(files_))

            # # if "add_predicted_files" value is given, it will be checked if it is already present in the overall old files list
            # # if present, error will be raised, else, it will be appended into old_files list and the input values list
            # if add_predicted_files:
            #     for i, file_name in enumerate(add_predicted_files):
            #         if file_name in old_files_list or file_name in all_old_files_list:
            #             raise ValueError(f"Duplicate file entry: {file_name}")
            #         else:

            #             old_files_list.append(file_name)
            #             files_.append(file_name)

            # # if there are any previously_added_files, they will also be appended to the input values list
            # try:
            #     for file_name in previously_added_files:
            #         files_.append(file_name)
            # except Exception:
            #     pass

            # # This will enable the transfer of files from the "files_predicted" section to "files_to_predict" section
            # # It is done by simply removing the file from "old_files_list"
            # if reverse_update_predicted_files:
            #     for file_name in reverse_update_predicted_files:
            #         old_files_list.remove(file_name)

            # # This will enable the transfer of files from "files_to_predict" section to "files_predicted" section
            # # This will update the list, "search_list" based on whether the parameter: "update_predicted_files" is given or not
            # if update_predicted_files:
            #     search_list = update_predicted_files + old_files_list
            # else:
            #     search_list = old_files_list

            # # This will iterate through input values list and check if the value in it is present in the "search_list"
            # for i in range(len(files_)):
            #     if files_[i] == TEST_DATA or files_[i] == TRAIN_DATA:
            #         file_name = "training_set" if files_[i] == TRAIN_DATA else "testing_set"
            #         files_from_s3_dict['files_for_model_training'][file_name] = files_[i]

            #     elif files_[i] in search_list:
            #         file_name = f"file_{predicted_files_counter}"
            #         files_from_s3_dict['files_predicted'][file_name] = files_[i]
            #         predicted_files_counter += 1

            #     else:
            #         file_name = f"file_{files_to_predict_counter}"
            #         files_from_s3_dict['files_to_predict'][file_name] = files_[i]
            #         files_to_predict_counter += 1

            # save_yaml(file=files_from_s3_dict,
            #           filepath=self.data_config.data_from_s3,
            #           mode='w')

        # This else block is the initial file creation of the file_lineage file from S3.
        # The files present in S3 will all be tracked under the  "files_to_predict" section intially.
        # Once the file_lineage file is created, the files(if any) that have been used for prediction can ->
        # -> be updated using the "update_predicted_files" parameter to move those file to the "files_predicted" section.
        # It will not contain the details of files entered through "Bulk/Batch" prediction.
        # The tracking of files entered through "Bulk/Batch" prediction is done only after this initial creation of file_lineage file.
        else:
            with open(self.data_config.data_from_s3, 'w'):
                files_from_s3_dict = {}
                files_from_s3_dict['files_for_model_training'] = {}
                files_from_s3_dict['files_to_predict'] = {}
                files_from_s3_dict['files_predicted'] = {}
                file_counter = 1
                for i in range(len(files_in_s3_)):
                    if files_in_s3_[i]['Key'] == TEST_DATA or files_in_s3_[i]['Key'] == TRAIN_DATA:
                        file_name = "training_set" if files_in_s3_[i]['Key'] == TRAIN_DATA else "testing_set"
                        files_from_s3_dict['files_for_model_training'][file_name] = files_in_s3_[i]['Key']
                    else:
                        file_name = f"S3_file_{file_counter}"
                        files_from_s3_dict['files_to_predict'][file_name] = files_in_s3_[i]['Key']
                        file_counter += 1
                save_yaml(file=files_from_s3_dict,
                          filepath=self.data_config.data_from_s3,
                          mode='w')

                # self.data_db_upload()


class data_db_uploader_component(file_tracker_component):
    def __init__(self) -> None:
        super().__init__()

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

        for config in [train_data_config, test_data_config]:
            self.s3_data_download(key=os.path.basename(config[0][0]),
                                  filepath=config[0][0])
            DB_data_uploader(config)

        shutil.rmtree(self.data_config.temp_dir_root)
        logger.info("Temporary directory removed")


# data_db_obj = data_db_uploader_component()
# data_db_obj.data_db_upload()
