from sklearn.ensemble import (StackingClassifier, VotingClassifier)
import warnings as w  # type: ignore
from typing import NewType  # type: ignore
from mlflow.client import MlflowClient
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.xgboost
import mlflow.sklearn
import mlflow
import optuna
from hyperopt import Trials, fmin, tpe, space_eval
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             accuracy_score, confusion_matrix)
from sklearn.pipeline import Pipeline as sk_pipeline
from imblearn.pipeline import Pipeline as imb_pipeline
from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import RFE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,  # noqa
                              GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)  # noqa
from sklearn.tree import DecisionTreeClassifier  # noqa
from xgboost import XGBClassifier  # noqa
from lightgbm import LGBMClassifier  # noqa
from sklearn.neighbors import KNeighborsClassifier  # noqa
from sklearn.datasets import make_classification
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
import json
from src.constants import PARAMS_PATH  # , SCHEMA_PATH
from src import logger
from collections import Counter
from cryptography.fernet import Fernet
import yaml
import joblib
import re  # type: ignore
import os
import glob  # type: ignore
import shutil  # type: ignore
import subprocess  # type: ignore
# from box import ConfigBox
from pathlib import Path  # type: ignore
import numpy as np
import pandas as pd
exp_count = 293
ML_Model = NewType('Machine_Learning_Model', object)

w.filterwarnings('ignore')


def crypter(encrypt_or_decrypt: str, file_name: str, data_to_encrypt: dict = None):
    """
        - Function to encrypt and decrypt.
        - The keys are saved in Secrets/Keys
        - The encrypted files are saved in Secrets/Secrets

        Parameters
        ----------
        encrypt_or_decrypt : str
            The type on action to perform. Accepts: 'encrypt' or 'decrypt'

        file_name : str
            The name of the file under which keys and the encrypted file will be stored

        data_to_encrypt: dict
            The dictionary of the data to be encrypted. Required for encryption only.
    """
    key_path = "Secrets/Keys/" + file_name + "_secrets.key"
    file_path = "Secrets/Secrets/" + file_name + "_config.yaml"

    if encrypt_or_decrypt == 'encrypt':
        # Generate the unique for this data
        key = Fernet.generate_key()
        with open(key_path, 'wb') as key_file:
            key_file.write(key)
        cipher_encrypt = Fernet(key)

        encrypted_data = {}
        for key, value in data_to_encrypt.items():
            encrypted_data[key] = cipher_encrypt.encrypt(value.encode()).decode()

        # Save encrypted data to a configuration file
        save_yaml(file=encrypted_data,
                  filepath=file_path,
                  mode='w')

    elif encrypt_or_decrypt == 'decrypt':
        # Read the key_file
        try:
            with open(key_path, 'rb') as key_file:
                key = key_file.read()
            cipher_decrypt = Fernet(key)

            # Load encrypted data from the yaml file
            loaded_data = load_yaml(filepath=file_path)

            # Decrypt sensitive information
            decrypted_data = {}
            for key, value in loaded_data.items():
                decrypted_data[key] = cipher_decrypt.decrypt(value.encode()).decode()

            return decrypted_data

        except FileNotFoundError:
            print(f"Key file '{key_file}' not found!")
            logger.info(f"Key file '{key_file}' not found!")


def load_yaml(filepath: Path):
    try:
        filepath_, filename = os.path.split(filepath)
        with open(filepath) as yaml_file:
            config = yaml.load(yaml_file,
                               Loader=yaml.CLoader)
            logger.info(f"{filename} yaml_file is loaded")
            return config
    except Exception as e:
        raise e


def save_yaml(file=None, filepath: Path = None, mode: str = 'w'):
    try:
        yaml.dump(data=file,
                  stream=open(file=filepath, mode=mode),
                  indent=4)
        logger.info("yaml file is saved")
    except Exception as e:
        raise e


def load_binary(filepath: Path):
    try:
        object = joblib.load(filename=filepath)
        logger.info(f"pickled_object: {filepath} loaded")
        return object
    except Exception as e:
        raise e


def save_binary(file, filepath: Path):
    try:
        joblib.dump(file, filepath)
        logger.info(f"object: {filepath} pickled")
    except Exception as e:
        raise e


def make_synthetic_data_for_unit_testing():
    schema_1 = ['ident_id', 'aa_000', 'ab_000', 'ac_000', 'ad_000', 'ae_000',
                'af_000', 'ag_000', 'ag_001', 'ag_002', 'ag_003', 'ag_004',
                'ag_005', 'ag_006', 'ag_007', 'ag_008', 'ag_009', 'ah_000',
                'ai_000', 'aj_000', 'ak_000', 'al_000', 'am_0', 'an_000',
                'ao_000', 'ap_000', 'aq_000', 'ar_000', 'as_000', 'at_000',
                'au_000', 'av_000', 'ax_000', 'ay_000', 'ay_001', 'ay_002',
                'ay_003', 'ay_004', 'ay_005', 'ay_006', 'ay_007', 'ay_008',
                'ay_009', 'az_000', 'az_001', 'az_002', 'az_003', 'az_004',
                'az_005', 'az_006', 'az_007', 'az_008', 'az_009', 'ba_000',
                'ba_001', 'ba_002', 'ba_003', 'ba_004', 'ba_005', 'ba_006',
                'ba_007', 'ba_008', 'ba_009', 'bb_000', 'bc_000', 'bd_000',
                'be_000', 'bf_000', 'bg_000', 'bh_000', 'bi_000', 'bj_000',
                'bk_000', 'bl_000', 'field_74_']
    schema_2 = ['ident_id', 'bm_000', 'bn_000', 'bo_000', 'bp_000', 'bq_000',
                'br_000', 'bs_000', 'bt_000', 'bu_000', 'bv_000', 'bx_000',
                'by_000', 'bz_000', 'ca_000', 'cb_000', 'cc_000', 'cd_000',
                'ce_000', 'cf_000', 'cg_000', 'ch_000', 'ci_000', 'cj_000',
                'ck_000', 'cl_000', 'cm_000', 'cn_000', 'cn_001', 'cn_002',
                'cn_003', 'cn_004', 'cn_005', 'cn_006', 'cn_007', 'cn_008',
                'cn_009', 'co_000', 'cp_000', 'cq_000', 'cr_000', 'cs_000',
                'cs_001', 'cs_002', 'cs_003', 'cs_004', 'cs_005', 'cs_006',
                'cs_007', 'cs_008', 'cs_009', 'ct_000', 'cu_000', 'cv_000',
                'cx_000', 'cy_000', 'cz_000', 'da_000', 'db_000', 'dc_000',
                'dd_000', 'de_000', 'df_000', 'dg_000', 'dh_000', 'di_000',
                'dj_000', 'dk_000', 'dl_000', 'dm_000', 'dn_000', 'do_000',
                'dp_000', 'dq_000', 'dr_000']
    schema_3 = ['ident_id', 'ds_000', 'dt_000', 'du_000', 'dv_000', 'dx_000',
                'dy_000', 'dz_000', 'ea_000', 'eb_000', 'ec_00', 'ed_000',
                'ee_000', 'ee_001', 'ee_002', 'ee_003', 'ee_004', 'ee_005',
                'ee_006', 'ee_007', 'ee_008', 'ee_009', 'ef_000', 'eg_000']
    schema_1 + schema_2 + schema_3
    sample_df = []
    for i in range(3):
        x, y = make_classification(n_features=len(
            eval(f"schema_{i+1}")), n_samples=180, random_state=42)
        print(x.shape)
        print(eval(f"schema_{i+1}"))
        df = pd.DataFrame(x, columns=eval(f"schema_{i+1}"))
        df['ident_id'] = list(range(1, 181))
        if i == 0:
            df[df.drop(columns=['ident_id', 'field_74_']) < -1] = 'na'
            df['field_74_'] = y
            df['field_74_'] = df['field_74_'].map({0: 'neg', 1: 'pos'})
        else:
            df[df.drop(columns=['ident_id']) < -1] = 'na'
        # df=stage_1_processing_function(df)
        sample_df.append(df)
    return sample_df


def batch_counter(dict_: dict):
    counter = 0
    for key in dict_.keys():
        if key.startswith('Batch'):
            counter += 1
    return counter


def s3_counter(dict_: dict):
    counter = 0
    for key in dict_.keys():
        if key.startswith('S3'):
            counter += 1
    return counter


def get_files_list_in_s3(files_dict: dict):
    files_from_s3_dict = {}
    for i in range(len(files_dict)):
        files_from_s3_dict[f"file_{i+1}"] = files_dict[i]['Key']
    return list(files_from_s3_dict.values())


def file_lineage_s3_files_refresher(files_in_s3: dict, old_files: dict):

    files_from_s3_list = get_files_list_in_s3(files_in_s3)

    old_files_for_model_training = old_files['files_for_model_training']
    old_files_predicted = old_files['files_predicted']
    old_files_to_predict = old_files['files_to_predict']

    for key, value in old_files_for_model_training.items():
        if value in files_from_s3_list:
            files_from_s3_list.remove(value)
        else:
            old_files_for_model_training.pop(key)
            raise ValueError(f"{value} is missing from S3")

    files_from_s3_list_ = files_from_s3_list.copy()
    key_list = []
    for key, value in old_files_predicted.items():
        if key.startswith("S3"):
            if value in files_from_s3_list_:
                files_from_s3_list.remove(value)
            else:
                key_list.append(key)
    if len(key_list) > 0:
        for key in key_list:
            old_files_predicted.pop(key)

    key_list = []
    files_from_s3_list_ = files_from_s3_list.copy()
    for key, value in old_files_to_predict.items():
        if key.startswith("S3"):
            if value in files_from_s3_list_:
                files_from_s3_list.remove(value)
            else:
                key_list.append(key)
    if len(key_list) > 0:
        for key in key_list:
            old_files_to_predict.pop(key)

    if len(files_from_s3_list) > 0:
        s3_count = s3_counter(old_files_to_predict)
        for i, file in enumerate(files_from_s3_list):
            old_files_to_predict[f'S3_file_{s3_count+(i+1)}'] = file

    old_files['files_for_model_training'] = old_files_for_model_training
    old_files['files_predicted'] = old_files_predicted
    old_files['files_to_predict'] = old_files_to_predict

    return old_files


def file_lineage_adder(add_list: list, old_files_predicted: dict):
    batch_count = batch_counter(old_files_predicted)
    for i, file in enumerate(add_list, start=1):
        if file in list(old_files_predicted.values()):
            raise ValueError(f"Duplicate file entry: {file}")
        else:
            old_files_predicted[f"Batch_file_{batch_count+(i)}"] = file
    return old_files_predicted


def file_lineage_updater(update_list: list, old_files_predicted: dict, old_files_to_predict: dict):
    key_list = []
    for i in update_list:
        for key, value in old_files_to_predict.items():
            if i == value:
                if key.startswith("S3"):
                    key_list.append(key)
                    s3_count = s3_counter(old_files_predicted)
                    old_files_predicted[f"S3_file_{s3_count+1}"] = i
                elif key.startswith('Batch'):
                    key_list.append(key)
                    batch_count = batch_counter(old_files_predicted)
                    old_files_predicted[f"Batch_file_{batch_count+1}"] = i
    for i in key_list:
        old_files_to_predict.pop(i)

    s3_count = s3_counter(old_files_to_predict)
    batch_count = batch_counter(old_files_to_predict)
    remaining_s3_files_to_predict = []
    remaining_batch_files_to_predict = []
    for key in old_files_to_predict.keys():
        if key.startswith("S3"):
            remaining_s3_files_to_predict.append(old_files_to_predict[key])
        elif key.startswith("Batch"):
            remaining_batch_files_to_predict.append(old_files_to_predict[key])
    old_files_to_predict = {}
    try:
        for i in range(s3_count):
            old_files_to_predict[f"S3_file_{i+1}"] = remaining_s3_files_to_predict[i]
    except Exception:
        pass
    try:
        for i in range(batch_count):
            old_files_to_predict[f"Batch_file_{i+1}"] = remaining_batch_files_to_predict[i]
    except Exception:
        pass

    return (old_files_predicted, old_files_to_predict)


def file_lineage_reverse_updater(reverse_update_list: list, old_files_predicted: dict, old_files_to_predict: dict):
    key_list = []
    for i in reverse_update_list:
        for key, value in old_files_predicted.items():
            if i == value:
                if key.startswith("S3"):
                    key_list.append(key)
                    s3_count = s3_counter(old_files_to_predict)
                    old_files_to_predict[f"S3_file_{s3_count+1}"] = i
                elif key.startswith('Batch'):
                    key_list.append(key)
                    batch_count = batch_counter(old_files_to_predict)
                    old_files_to_predict[f"Batch_file_{batch_count+1}"] = i
    for i in key_list:
        old_files_predicted.pop(i)

    s3_count = s3_counter(old_files_predicted)
    batch_count = batch_counter(old_files_predicted)
    remaining_s3_files = []
    remaining_batch_files = []
    for key in old_files_predicted.keys():
        if key.startswith("S3"):
            remaining_s3_files.append(old_files_predicted[key])
        elif key.startswith("Batch"):
            remaining_batch_files.append(old_files_predicted[key])
    old_files_predicted = {}
    try:
        for i in range(s3_count):
            old_files_predicted[f"S3_file_{i+1}"] = remaining_s3_files[i]
    except Exception:
        pass
    try:
        for i in range(batch_count):
            old_files_predicted[f"Batch_file_{i+1}"] = remaining_batch_files[i]
    except Exception:
        pass
    return (old_files_predicted, old_files_to_predict)


def DB_data_uploader(config: list):
    main_data_path = config[0][0]
    logger.info(f"{main_data_path} is the main data path")

    file_path, file = os.path.split(main_data_path)
    df = pd.read_csv(main_data_path)
    # if file == 'remote_aps_failure_training_set.csv':
    #     range_ = 60001
    #     logger.info(f"{range_} is the range")
    # else:
    #     range_ = 16001
    #     logger.info(f"{range_} is the range")
    range_ = len(df) + 1
    logger.info(f"{range_} is the range")

    df_1 = df.iloc[:, :74]
    df_1['ident_id'] = range(1, range_)
    df_1.to_csv(config[1][4], index=False)
    logger.info(f"{file.split('.')[0]}_1 saved in temp folder")

    df_2 = df.iloc[:, 74:148]
    df_2['ident_id'] = range(1, range_)
    df_2.to_csv(config[2][4], index=False)
    logger.info(f"{file.split('.')[0]}_2 saved in temp folder")

    df_3 = df.iloc[:, 148:]
    df_3['ident_id'] = range(1, range_)
    df_3.to_csv(config[3][4], index=False)
    logger.info(f"{file.split('.')[0]}_3 saved in temp folder")

    for i in list(range(1, 4)):
        data_db_config = config[i]
        token = data_db_config[1]
        logger.info("Token loaded")
        # print(token)

        with open(token) as json_file:
            secrets = json.load(json_file)
            # logger.info(f"{config[1]} json file is loaded")

        secure_connect_bundle = data_db_config[0]
        logger.info("Secure Connect Bundle loaded")
        # print(secure_connect_bundle)

        cloud_config = {'secure_connect_bundle': secure_connect_bundle,
                        'connect_timeout': None}
        # print(f"cloud_config: {cloud_config}")
        CLIENT_ID = secrets["clientId"]
        # print(CLIENT_ID)
        CLIENT_SECRET = secrets["secret"]
        # print(CLIENT_SECRET)
        auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
        cluster = Cluster(cloud=cloud_config,
                          auth_provider=auth_provider,
                          protocol_version=4)
        session = cluster.connect()
        logger.info("Cluster connected")

        data_df_path = data_db_config[4]
        data_df = pd.read_csv(data_df_path)
        # print(data_df)

        # Define keyspace name and table name
        keyspace = data_db_config[2]
        table_name = data_db_config[3]
        logger.info("Keyspace & Table_name defined")
        # print("Keyspace", keyspace)
        # print("Table_name", table_name)

        colums_types_dict_from_data_df = dict(zip(list(data_df.columns), list(data_df.dtypes.replace({'object': 'text', 'int64': 'int'}).values)))
        columns = ', '.join([f"{key} {value}" for key, value in colums_types_dict_from_data_df.items()])
        create_table_statement = f"CREATE TABLE IF NOT EXISTS {keyspace}.{table_name} ({columns}, PRIMARY KEY (ident_id));"
        session.execute(create_table_statement)
        logger.info("Table created")

        dsbulk_command = [
            "dsbulk",
            "load",
            "-url", data_df_path,
            "-k", keyspace,
            "-t", table_name,
            "-b", secure_connect_bundle,
            "-u", CLIENT_ID,
            "-p", CLIENT_SECRET,
        ]
        result = subprocess.run(dsbulk_command, shell=True)
        file_path, file_ = os.path.split(data_df_path)
        logger.info(f"{file_} uploaded successfully")
        print(f"Return code: {result.returncode}")
        print(f"Standard output: {result.stdout}")
        print(f"Standard error: {result.stderr}\n\n")

        session.shutdown()
        cluster.shutdown()

    logger.info(f"The entire {file} is uploaded successfully in batches.")

    # Removal of temporary data files
    for i in range(1, 4):
        os.remove(config[i][4])
        logger.info(f"Removed : {config[i][4]}")
    os.remove(config[0][0])
    logger.info(f"Removed : {config[0][0]}")

    # Removal of log files created by DSbulk
    directory_path = "logs"
    pattern = os.path.join(directory_path, "LOAD*")
    files_to_remove = glob.glob(pattern)
    for file_path in files_to_remove:
        shutil.rmtree(file_path)
        logger.info(f"Removed : {file_path}")


def DB_data_downloader(config: list):
    with open(config[1]) as f:
        secrets = json.load(f)
        logger.info(f"{config[1]} json file is loaded")

    cloud_config = {'secure_connect_bundle': config[0],
                    'connect_timeout': None}
    print(f"cloud_config: {cloud_config}")
    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]
    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    profile = ExecutionProfile(request_timeout=None)
    cluster = Cluster(cloud=cloud_config,
                      auth_provider=auth_provider,
                      execution_profiles={EXEC_PROFILE_DEFAULT: profile},
                      protocol_version=4)

    session = cluster.connect()

    logger.info(f"connected to cluster/keyspace: {config[2]}")

    keyspace = config[2]
    table = config[3]
    query = f"SELECT * FROM {keyspace}.{table}"

    result = session.execute(query)
    logger.info(f"Data queried from keyspace: {config[2]}")
    df = pd.DataFrame(list(result))
    df.to_csv(config[4], index=False)
    print(df.shape)
    session.shutdown()
    cluster.shutdown()
    return (auth_provider.username, auth_provider.password)


def stage_1_processing_function(dataframes: list) -> pd.DataFrame:
    logger.info("Stage 1 Processing Commencing")

    data_merger_1 = pd.merge(left=dataframes[0],
                             right=dataframes[1],
                             how='outer',
                             on='ident_id')
    logger.info("Data Merging commencing")

    data_merger_final = pd.merge(left=data_merger_1,
                                 right=dataframes[2],
                                 how='outer',
                                 on='ident_id')
    logger.info("Data Merging complete")

    data_merger_final = data_merger_final.sort_values(by='ident_id').reset_index(drop=True)
    logger.info("Sorting and reseting_index complete")

    data_merger_final.drop(columns='ident_id', inplace=True)
    logger.info("Dropping column: 'ident_id'")

    data_merger_final.rename(columns={'field_74_': 'class'}, inplace=True)
    logger.info("Renaming Target Column")

    data_merger_final_ = data_validation_helper(data_merger_final)

    logger.info("Stage 1 processing complete - Returning processed dataframe")
    return (data_merger_final_)


def schema_saver(dataframe: pd.DataFrame,  labels: list, mode: str, filepath=None):
    if len(labels) > 2:
        raise ValueError("The 'labels' argument should not have more than 2 values.")
    from src.config.configuration_manager import ConfigurationManager
    obj = ConfigurationManager()
    dict_cols = {}
    flag = 0
    try:
        labels.remove('Target')
        flag = 1
    except Exception:
        pass
    for i in labels:
        dict_cols[i] = {}
    for i in dataframe.columns:
        if i == 'class':
            if flag == 1:
                dict_cols['Target'] = {}
                dict_cols['Target'][i] = str(dataframe[i].dtypes)
        else:
            dict_cols[labels[0]][i] = str(dataframe[i].dtypes)
    if not filepath:
        filepath = obj.schema
    save_yaml(file=dict_cols, filepath=filepath, mode=mode)


def train_test_splitter(dataframe: pd.DataFrame) -> pd.DataFrame:
    train, test = train_test_split(dataframe, test_size=0.25, random_state=8)
    return (train, test)


def stage_2_processing_function(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
        - In this final processing function; The NA values are imputed using KNN Imputer.
        - Further, the dataframe is scaled using Robust Scaler (as there are outliers)
        - Lastly, in this function, duplicate entries/rows are also removed
    """
    from src.config.configuration_manager import ConfigurationManager
    obj = ConfigurationManager()
    preprocessor_config = obj.get_preprocessor_config()
    schema = load_yaml(obj.schema)
    target = list(schema.Target.keys())[0]
    if not os.path.exists(preprocessor_config.preprocessor_path):
        logger.info("Stage 2 Processing Commencing for train_data")

        # logger.info("Commencing Data Validation")
        # dataframe_train = data_validation(dataframe=dataframe)

        logger.info("Saving schema with columns that, have less than 50% missing values and have > zero standard deviation")
        schema_saver(dataframe=dataframe,
                     labels=['Required_Features'],
                     mode='a')

        pipeline = sk_pipeline(steps=[('Knn_imputer', KNNImputer()),
                                      ('Robust_Scaler', RobustScaler())],
                               verbose=True)
        smote = SMOTETomek(
            n_jobs=-1, sampling_strategy='minority', random_state=42)

        logger.info("Pipeline created with KnnImputer, RobustScaler")
        logger.info("SmoteTomek obj created")

        X = dataframe.drop(columns=target)
        # logger.info(f"Creating X dataframe with only the input features - dropping target")

        y = dataframe[target]
        # logger.info(f"Creating y - target")

        logger.info("Commencing pipeline transformation")
        X_transformed = pipeline.fit_transform(X=X)
        logger.info("Pipeline transformation complete")

        logger.info("Commencing SmoteTomek")
        X_smote, y_smote = smote.fit_resample(X=X_transformed,
                                              y=y)
        logger.info("SmoteTomek Complete")

        columns_list = list(pipeline.get_feature_names_out())
        logger.info(f"Pipeline is fitted with the columns: {columns_list}")
        X_column_names = [i for i in columns_list if i != target]
        y_column_name = target

        logger.info("Returning the transformed dataframe")
        transformed_df = pd.DataFrame(X_smote, columns=X_column_names)
        transformed_df[y_column_name] = y_smote

        logger.info("Saving the pipeline object")
        save_binary(file=pipeline,
                    filepath=preprocessor_config.preprocessor_path)
        logger.info(
            f"Pipeline saved at: {preprocessor_config.preprocessor_path}")

        # logger.info("Loading histogram features")
        # hist_features = load_yaml(SCHEMA_PATH)['histogram_columns']
        # hist_features = (hist_features.keys())

        # logger.info("Commencing Feature Selection of histogram features")
        # hist_feature_selector = RFE(estimator = RandomForestClassifier(n_estimators=108,
        #                                                                n_jobs = -1,
        #                                                                random_state = 8),
        #                             verbose = 3)
        # hist_feature_selector.fit(transformed_df[hist_features],
        #                           transformed_df['class'])
        # hist_feature_selector.get_feature_names_out()

        # Checking for any duplicates
        if any(transformed_df.duplicated()):
            logger.info("Dropping duplicate rows/entries")
            shape = transformed_df[transformed_df.duplicated()].shape[0]
            transformed_df.drop_duplicates(inplace=True)
            logger.info(f"Number of duplicate rows/entries dropped {shape}")

        logger.info("Stage 2 Processing Complete")

        return (transformed_df)

    else:
        logger.info("Stage 2 Processing Commencing")

        loaded_pipeline = load_binary(preprocessor_config.preprocessor_path)
        smote = SMOTETomek(
            n_jobs=-1, sampling_strategy='minority', random_state=8)

        logger.info("Pipeline loaded & SmoteTomek created")

        X = dataframe.drop(columns=target)
        y = dataframe[target]

        logger.info("Commencing pipeline transformation")
        X_transformed = loaded_pipeline.transform(X=X)
        logger.info("Pipeline transformation complete")

        logger.info("Commencing SmoteTomek")
        X_smote, y_smote = smote.fit_resample(X=X_transformed,
                                              y=y)
        logger.info("SmoteTomek Complete")

        columns_list = list(loaded_pipeline.get_feature_names_out())
        X_column_names = [i for i in columns_list if i != target]
        y_column_name = target

        logger.info("Returning the transformed dataframe")
        transformed_df = pd.DataFrame(X_smote, columns=X_column_names)
        transformed_df[y_column_name] = y_smote

        # Checking for any duplicates
        if any(transformed_df.duplicated()):
            logger.info("Dropping duplicate rows/entries")
            shape = transformed_df[transformed_df.duplicated()].shape[0]
            transformed_df.drop_duplicates(inplace=True)
            logger.info(f"Number of duplicate rows/entries dropped {shape}")

        logger.info("Stage 2 Processing Complete")

        return (transformed_df)


def data_validation_helper(data_frame: pd.DataFrame):
    """
        This validation function:
            * Checks and:
                - Maps 'neg' to 0 and 'pos' to 1 in the given pandas dataframe
                - Converts 'na' to np.nan
                - Coverts the datatype of all the columns except target column to "float"
    """
    try:
        data_frame['class'] = data_frame['class'].map({'neg': 0, 'pos': 1})
        logger.info("Mapping Target Column values")
    except Exception:
        pass

    try:
        data_frame.replace('na', np.nan, inplace=True)
        logger.info("Replacing 'na' to 'np.nan' values")
    except Exception:
        pass

    test_col_list = [i for i in data_frame.columns if i != 'class']
    if len(test_col_list) != 170:
        raise ValueError("The length of dataframe's input features is not equal to 170")
    else:
        logger.info("Creating list of column names of input features")

    try:
        for i in test_col_list:
            data_frame[i] = data_frame[i].astype('float')
        logger.info("dtype of input features converted from 'object' to 'float'")
    except Exception as e:
        raise e

    return (data_frame)


def data_validation(dataframe: pd.DataFrame, cols_to_remove: list = None, columns_with_0_std_dev: list = None):
    """
        This validation function:
            * Checks and removes:
                - Columns that have more than 50% missing values (removed columns are saved in SCHEMA)
                - Columns that have 0 standard deviation (removed columns are saved in SCHEMA)
            * Checks and saves the histogram features present in the dataframe (histogram features are saved in SCHEMA)
    """
    # Checking for columns that have more than 50% missing values
    if cols_to_remove is None:
        logger.info("Fetching columns that have more than 50% missing values")
        missing_values_series = pd.Series(dataframe.isna().sum().sort_values(ascending=False)*100/dataframe.shape[0])
        missing_values_series = missing_values_series[missing_values_series >= 50]
        cols_to_remove = missing_values_series.index

        logger.info("Saving Schema with dropped columns that had more than 50% missing values under the name: 'columns_with_more_than_50%_missing_values")
        schema_saver(dataframe=dataframe[cols_to_remove],
                     labels=['columns_with_more_than_50%_missing_values'],
                     mode='a')

        logger.info("Dropping columns that have more than 50% missing values")
        dataframe.drop(columns=cols_to_remove, inplace=True)

        logger.info(f"Dropped columns and their respective values:\n{missing_values_series}")
    else:
        dataframe = data_validation_helper(dataframe)
        logger.info("Dropping same columns in test_data, that had more than 50% missing values in train_data")
        dataframe.drop(columns=cols_to_remove, inplace=True)
        logger.info(f"Dropped columns:\n{cols_to_remove}")

    # Checking for columns with zero standard deviation and dropping those columns
    if columns_with_0_std_dev is None:
        columns_with_0_std_dev = list(dataframe.std()[dataframe.std() == 0].index)

        logger.info("Saving Schema with columns that have zero standard deviation under: 'columns_with_zero_standard_deviation'")
        schema_saver(dataframe=dataframe[columns_with_0_std_dev],
                     labels=['columns_with_zero_standard_deviation'],
                     mode='a')

        dataframe.drop(columns=columns_with_0_std_dev, inplace=True)
        logger.info("Dropping columns with zero standard deviation")
        logger.info(f"Dropped columns are:\n{columns_with_0_std_dev}")

        # Checking for histogram features
        prefix = []
        for name in dataframe.columns:
            prefix.append(name.split('_')[0])
        counter_dict = Counter(prefix)
        hist_dict_ = {key: value for key, value in counter_dict.items() if value > 1}
        hist_features = [column for column in dataframe.columns if column.split('_')[0] in list(hist_dict_.keys())]

        logger.info("Saving Schema with histogram-features")
        schema_saver(dataframe=dataframe[hist_features],
                     labels=['histogram_columns'],
                     mode='a')
    else:
        logger.info("Dropping same columns in test_data, that had zero standard deviation in train_data")
        dataframe.drop(columns=columns_with_0_std_dev, inplace=True)
        logger.info(f"Dropped columns:\n{columns_with_0_std_dev}")

    return (dataframe)


def eval_metrics(y_true, y_pred):
    balanced_accuracy_score_ = float(
        balanced_accuracy_score(y_true=y_true, y_pred=y_pred))

    f1_score_ = float(f1_score(y_true=y_true, y_pred=y_pred))

    accuracy_score_ = float(accuracy_score(y_true=y_true, y_pred=y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    cost_ = float((10*fp)+(500*fn))

    return ({"Balanced_Accuracy_Score": balanced_accuracy_score_,
             "F1_Score": f1_score_,
             "Accuracy_Score": accuracy_score_,
             "Cost": cost_})


# DEPRECATED
def parameter_tuning(model_class,
                     model_name: str,
                     x_train: pd.DataFrame,
                     x_test: pd.DataFrame,
                     y_train: pd.DataFrame,
                     y_test: pd.DataFrame,
                     report_: dict,
                     client: MlflowClient,
                     *args):

    tuner_report = {}
    tuner_report['Optuna'] = {}
    tuner_report['HyperOpt'] = {}
    params_config = load_yaml(PARAMS_PATH)
    exp_id_list = []

    tags = {"tuner_1": "optuna",
            "tuner_2": "hyperopt",
            "metrics": "['Balanced_Accuracy_Score', 'F1_Score', 'Accuracy_Score', 'Cost']"}
    exp_id = client.create_experiment(name=f"63_{model_name}_63", tags=tags)

# OPTUNA
    with mlflow.start_run(experiment_id=exp_id,
                          run_name=f"Optuna for {model_name}",
                          tags={"tuner": "optuna",
                                "run_type": "parent"}) as optuna_parent_run:
        parent_run_id = optuna_parent_run.info.run_id

        def optuna_objective(trial):
            with mlflow.start_run(experiment_id=exp_id,
                                  run_name=f"Trial {(trial.number)+1} for {model_name} (optuna)",
                                  tags={"run_type": "child"},
                                  nested=True):
                space_optuna = {}
                for key, value in params_config['optuna'][model_name].items():
                    space_optuna[key] = eval(value)
                if model_name == 'Stacked_Classifier':
                    model = model_class.set_params(**space_optuna)
                else:
                    model = model_class(**space_optuna)
                # model.set_params(**space_optuna)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                cost = eval_metrics(y_test, y_pred)["Cost"]

                data = (x_train, y_test, y_pred)
                mlflow_logger(data=data,
                              model=model,
                              model_name=model_name,
                              #   params=model.get_params(),
                              should_log_parent_model=False,
                              artifact_path=f'optuna_{model_name}' if model_name == 'XGB_Classifier' else f'optuna_{model_name}')
                print("Artifacts URI of Optuna Child Run: ",
                      mlflow.get_artifact_uri())
                return cost

        print("Artifacts URI of Optuna Parent Run: ", mlflow.get_artifact_uri())
        find_param = optuna.create_study(direction="minimize")
        find_param.optimize(optuna_objective, n_trials=2)

        data = (x_train, x_test, y_train, y_test)
        mlflow_logger(data=data,
                      model_name=model_name,
                      should_log_parent_model=True,
                      run_id=parent_run_id,
                      exp_id=exp_id,
                      #   registered_model_name=f"Challenger_Optuna_{model_name}",
                      artifact_path=f'challenger_optuna_{model_name}' if model_name == 'XGB_Classifier' else f'challenger_optuna_{model_name}')

        tuner_report['Optuna'] = {
            'Cost': find_param.best_value, 'params': find_param.best_params}
        print(f"Optuna: {model_name} --- {tuner_report['Optuna']}\n\n")

# HYPEROPT
    with mlflow.start_run(experiment_id=exp_id,
                          run_name=f"HyperOpt for {model_name}",
                          tags={"tuner": "hyperopt",
                                "run_type": "parent"}) as hyperopt_parent_run:
        parent_run_id = hyperopt_parent_run.info.run_id
        global trial_number
        # trial_number=0

        def hp_objective(space):
            global trial_number
            trial_number += 1
            with mlflow.start_run(experiment_id=exp_id,
                                  run_name=f"Trial {trial_number} for {model_name} (hyperopt)",
                                  tags={"run_type": "child"},
                                  nested=True):

                if model_name == 'Stacked_Classifier':
                    model = model_class.set_params(**space)
                else:
                    model = model_class(**space)
                # model.set_params(**space)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                cost = eval_metrics(y_test, y_pred)["Cost"]
                print("Cost: ", cost)

                data = (x_train, y_test, y_pred)
                mlflow_logger(data=data,
                              model=model,
                              model_name=model_name,
                              should_log_parent_model=False,
                              artifact_path=f'hyperopt_{model_name}' if model_name == 'XGB_Classifier' else f'hyperopt_{model_name}')
                print("Artifacts URI of HyperOpt Child Run: ",
                      mlflow.get_artifact_uri())
                return cost
        print("Artifacts URI of HyperOpt Parent Run: ",
              mlflow.get_artifact_uri())
        trials = Trials()
        space = {}
        for key, value in params_config['hyperopt'][model_name].items():
            space[key] = eval(value)
        best = fmin(fn=hp_objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=2,
                    trials=trials)
        best_params = space_eval(space, best)

        data = (x_train, x_test, y_train, y_test)
        mlflow_logger(data=data,
                      model_name=model_name,
                      should_log_parent_model=True,
                      run_id=parent_run_id,
                      exp_id=exp_id,
                      #   registered_model_name=f"Challenger_HyperOpt_{model_name}",
                      artifact_path=f'challenger_hyperopt_{model_name}' if model_name == 'XGB_Classifier' else f'challenger_hyperopt_{model_name}')

        tuner_report['HyperOpt'] = {'Cost': int(
            trials.average_best_error()), 'params': best_params}
        print(f"HyperOpt: {model_name} --- {tuner_report['HyperOpt']}\n\n")
        # trial_number = 0

# Best_COST & Best_Fittable_Params
    min_cost_value = min(
        tuner_report['Optuna']['Cost'], tuner_report['HyperOpt']['Cost'])
    if min_cost_value == tuner_report['Optuna']['Cost']:
        params = tuner_report['Optuna']['params']
    else:
        params = tuner_report['HyperOpt']['params']
    tuner_report['Fittable_Params'] = params
    tuner_report['Best_Cost'] = min_cost_value

    report_[model_name] = tuner_report
    print(f'\n\n{model_name}\nMin Cost: {min_cost_value}\n{report_[model_name]}\n\n')
    # print(report_.values())
    costs = [value['Best_Cost'] for value in report_.values()]
    min_cost = min(costs)
    best_model_so_far_ = [(i, min_cost, report_[i]['Fittable_Params'])
                          for i in report_.keys()
                          if min_cost == report_[i]['Best_Cost']]

    data = x_train
    mlflow_logger(data=data,
                  model_name=model_name,
                  #   should_register_model=True,
                  exp_id=exp_id,
                  registered_model_name=f"Challenger_{model_name}",
                  artifact_path=None)
    exp_id_list.append(exp_id)

    return (tuner_report, report_, best_model_so_far_, exp_id_list)
# DEPRECATED


def parameter_tuning_2(models: dict, client: MlflowClient,
                       dataframe: pd.DataFrame, optuna_trials_study_df_path: Path):
    """
    Parameter Tuning for batch-wise data.
    cross_validate() is used to handle the batch-wise data.

    Optuna is used to tune and select the optimal
    hyperparameters of a given model.

    The batch data is passed into this function along
    with the models list that we want to check.

    This function implements MLFlow tracking to log the metrics and params.
    This function also creates an experiment in MLFlow for
    each and every model.

    Inside the experiment of a model in MLFlow, a parent run is created with
    Optuna selecting a set of Hyper-Parameters for the model.
    Inside the context of this parent run, for a given model, every iteration
    of a given batch of data, for a selected hyper-parameter set, will be
    logged as a child run.

    The batches(or child runs) will continue iterating normally till first
    five batches, and then from the sixth batch onwards and for every even
    batch number, the 25th quantile of the accuracies of the previous batches will be checked with
    the current batches' accuracy. If the current batch's accuracy is lower
    than the 25th quantile accuracy of previous batches, that trial(consisting of the
    parent run and all its child runs) will be pruned.

    25th quantile is set as the threshold to allow some liniency otherwise the pruning check will be too strict and
    as a result most trials will be pruned.

    Those iterations that continue till the very last batch will be logged in
    MLFlow as child runs.
    From the performance of child runs, the best performing child run is
    chosen as the parent run for that trial.

    Like this, 3 trials in optuna are done, where child and parent runs are
    created and logged if they iterate till the last batch of data.

    After the 3rd trial, all the parent runs in that model's experiment are
    compared and the best parent is chosen as the Best_HP_Tuned_Model.
    """
    from pprint import pprint  # type: ignore

    batch_list = batch_data_create(dataframe)

    exp_id_list = []
    params = load_yaml(PARAMS_PATH)
    model_trial_study_df = pd.DataFrame()
    accuracies = {}
    for keys in models.keys():
        accuracies[keys] = {}
        accuracies[keys]['Optuna'] = []
        # accuracies[keys]['HyperOpt']=[]
    tags = {"tuner": "optuna",
            "metrics": "['Balanced_Accuracy_Score', 'F1_Score', 'Accuracy_Score', 'Cost']"}
    count = 1
    for key, value in models.items():
        exp_id = mlflow.create_experiment(name=f"{exp_count}_{key}_{exp_count}", tags=tags)
        # if key == 'Bagging_Classifier':
        #     light_gbm = best_model_finder(model_name='Light_GBM', search_registered_models=False)

        def optuna_objective(trial, exp_id=exp_id):
            with mlflow.start_run(experiment_id=exp_id,
                                  run_name=f"{key}_Trial_{trial.number}",
                                  tags={"tuner": "optuna",
                                        'trial': f"{trial.number}",
                                        'model': key,
                                        "run_type": "parent"}) as parent_run:
                parent_run_id = parent_run.info.run_id
                child_id_list = []
                batch_wise_accuracy = []
                space = {}
                flag = 0
                print(
                    f"\n*******\nTrial_Number: {trial.number}\n*******")
                for key_, value_ in params['optuna'][key].items():
                    space[key_] = eval(value_)
                # if key == 'Bagging_Classifier':
                #     space['estimator'] = light_gbm
                for i in range(len(batch_list)):  # CHANGED
                    x = batch_list[i].drop(columns='class')
                    y = batch_list[i]['class']
                    print(f"\nBatch {i}")
                    pprint(f"\nSpace: {space}", compact=True)
                    pipeline = imb_pipeline(steps=[("KNN_Imputer", KNNImputer()),
                                                   ("Robust_Scaler", RobustScaler()),
                                                   ("SMOTETomek", SMOTETomek(sampling_strategy="minority", random_state=42)),
                                                   (f"{key}", value(**space))])
                    skf_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    cv_results = cross_validate(estimator=pipeline,
                                                X=x,
                                                y=y,
                                                scoring='accuracy',
                                                cv=skf_cv,
                                                n_jobs=-1,
                                                verbose=2,
                                                return_estimator=True)
                    accuracy = cv_results['test_score']
                    pipeline = cv_results['estimator'][0]
                    estimator = pipeline.named_steps[key]
                    print(f"\nAccuracy of {key}: {np.mean(accuracy)}")

                    batch_wise_accuracy.append(np.mean(accuracy))
                    accuracies[key]['Optuna'].append(np.mean(accuracy))
                    if i > 5:
                        print(
                            f"Dynamic/Moving Threshold: {dynamic_threshold(all_accuracies=batch_wise_accuracy)}"
                        )
                    print(f"\nBatchwise_accuracies: {batch_wise_accuracy}")
                    print(
                        f"\nBatchwise_Median_Accuracy: {np.median(batch_wise_accuracy)}")
                    # print(f"\nAccumulated Accuracies:\n{accuracies}\n")

                    trial.report(np.mean(accuracy), i)
                    child_run_id = 0
                    with mlflow.start_run(experiment_id=exp_id,
                                          run_name=f"Batch_{i}",
                                          tags={"tuner": "optuna",
                                                'model': key,
                                                "run_type": "child"},
                                          nested=True) as child_run:
                        child_run_id = child_run.info.run_id
                        child_id_list.append(child_run_id)
                        mlflow_logger(model=estimator,
                                      client=client,
                                      model_name=key,
                                      should_log_parent_model=False,
                                      artifact_path=f'{key}_batch_{i}')
                        mlflow.log_metrics(
                            metrics={"Accuracy_Score": np.mean(accuracy)})
                        dataset_ = mlflow.data.from_pandas(df=batch_list[i],
                                                           targets='class',
                                                           name=f"Batch {i}")
                        mlflow.log_input(dataset=dataset_,
                                         context='Batchwise Cross-Validation')
                    if (i > 5) and (i % 2 != 0):  # CHANGED
                        trial.study.set_user_attr(
                            'median_accuracy', np.median(batch_wise_accuracy))
                        # np.median(batch_wise_accuracy):
                        if np.mean(accuracy) < dynamic_threshold(
                                all_accuracies=batch_wise_accuracy):
                            flag = 1
                            break
                if flag == 0:
                    mlflow_logger(model_name=key,
                                  should_log_parent_model=True,
                                  client=client,
                                  metrics_={"Accuracy_Score": np.mean(
                                      batch_wise_accuracy)},
                                  run_id=parent_run_id,
                                  exp_id=exp_id,
                                  artifact_path=f'candidate_{key}')
            if flag == 1:
                mlflow.delete_run(run_id=parent_run_id)
                try:
                    for child_run in child_id_list:
                        mlflow.delete_run(child_run)
                except Exception:
                    pass

                raise optuna.TrialPruned()

            return np.mean(batch_wise_accuracy)
        find_params = optuna.create_study(direction='maximize',
                                          pruner=optuna.pruners.MedianPruner(),
                                          sampler=optuna.samplers.TPESampler(constant_liar=True))
        find_params.optimize(func=optuna_objective,
                             n_trials=3)
        if count == 1:
            model_trial_study_df = find_params.trials_dataframe()
            model_trial_study_df['Model_name'] = key
            count += 1
        else:
            temp_df = find_params.trials_dataframe()
            temp_df['Model_name'] = key
            model_trial_study_df = pd.concat([model_trial_study_df, temp_df])
        model_trial_study_df.to_csv(optuna_trials_study_df_path, index=False)

        mlflow_df = mlflow.search_runs(experiment_ids=[f'{exp_id}'])
        if mlflow_df.empty:
            pass
        else:
            mlflow_logger(client=client,
                          model_name=key,
                          exp_id=exp_id,
                          registered_model_name=f"Candidate_{key}",
                          artifact_path=None)
            exp_id_list.append(exp_id)

    best_exp_id = mlflow_logger(exp_id=exp_id_list,
                                should_register_model=True,
                                client=client,
                                registered_model_name='Candidate_',
                                artifact_path=None)

    return (model_trial_study_df, exp_id_list, accuracies, best_exp_id)


def dynamic_threshold(all_accuracies, quantile=0.25):
    # Calculate the moving average trend
    recent_accuracies = all_accuracies[:-1]

    threshold = np.quantile(recent_accuracies, quantile)
    # moving_average=(recent_accuracies)

    # Calculate the threshold dynamically based on the moving average
    # threshold=recent_accuracies * threshold_factor
    return threshold


def best_model_finder(models: dict = None, client: MlflowClient = None, model_name: str = None,
                      filter_string=None, search_registered_models: bool = True):
    # DEPRECATED
    # best_models_=sorted(report.items(), key=lambda x: x[1]['Best_Cost'])[:7]
    # best_models=[(best_models_[i][0],report[best_models_[i][0]]['Best_Cost']) for i in range(len(best_models_))]
    # print('\nBest Models:')
    # for i in best_models:
    #     print(i[0]," Cost: ", i[1],'\n\n')
    # best_models_with_params=[]
    # for i in best_models:
    #     # print(f"i: {i[0]}")
    #     best_models_with_params.append((i[0],report[i[0]]['Fittable_Params']))
    # best_estimators={}
    # # print("report:\n",report)
    # for i in range(len(best_models_with_params)):
    #     # print ("best_models_with_params[i][0]: \n",best_models_with_params[i][0])
    #     if (best_models_with_params[i][0] == 'Stacked_Classifier'): #best_models_with_params[i][0] == 'Voting_Classifier'):
    #         best_estimators[best_models_with_params[i][0]]=models[best_models_with_params[i][0]]
    #         # best_estimators[best_models_with_params[i][0]].set_params(**best_models_with_params[i][1])

    #     elif (best_models_with_params[i][0] == 'Voting_Classifier'):
    #         best_estimators[best_models_with_params[i][0]]=models[best_models_with_params[i][0]]
    #         # best_estimators[best_models_with_params[i][0]].set_params()
    #     else:
    #         best_estimators[best_models_with_params[i][0]]=models[best_models_with_params[i][0]](**best_models_with_params[i][1])
    #         # best_estimators[best_models_with_params[i][0]].set_params(**best_models_with_params[i][1])
    # best_estimators=list(zip(best_estimators.keys(),best_estimators.values()))

    # costs=[value['Best_Cost'] for value in report.values()]
    # min_cost=min(costs)
    # best_model_so_far_=[(i, min_cost, report[i]['Fittable_Params']) for i in report.keys() if min_cost == report[i]['Best_Cost']]

    # return (best_model_so_far_,best_models_with_params,best_estimators)
    # DEPRECATED

    # Get the dictionary of all the registered challenger models from MLFlow.
    # This dict will have model names as keys and the run_IDs as values.
    if search_registered_models is False:
        best_run_id = mlflow.search_runs(experiment_names=[f"{exp_count}_{model_name}_{exp_count}"],
                                         filter_string=f"tags.model ilike '{model_name}' and tags.run_type ilike 'parent'",
                                         order_by=['metrics.Accuracy_Score DESC'])['run_id'][0]
        best_params = params_evaluator(mlflow.get_run(best_run_id).data.params)
        params = load_yaml(PARAMS_PATH)
        best_model = eval(params.models[f'{model_name}'])(**best_params)
        return (best_model)

    else:
        registered_models = {mlflow.search_registered_models(filter_string=filter_string)[i].latest_versions[0].name: mlflow.search_registered_models(
            filter_string=filter_string)[i].latest_versions[0].run_id for i in range(len(mlflow.search_registered_models(filter_string=filter_string)))}

        # Get the accuracies and the respective params of the models as dict
        run_details = {}
        for key, value in registered_models.items():
            run_details[client.get_run(value).data.tags['model']] = {}
            run_details[client.get_run(value).data.tags['model']]['accuracy'] = client.get_run(
                value).data.metrics['Accuracy_Score']
            run_details[client.get_run(value).data.tags['model']]['params'] = params_evaluator(
                client.get_run(value).data.params)

        # Create a dataframe from the "run_details" dict and sort it by "accuracy" in DESC
        # In this dataframe only the models whose accuracy is greater than 0.9 are chosen.
        models_df = pd.DataFrame(run_details).T

        sorted_models_df = models_df[models_df['accuracy'] > 0.9].sort_values(
            by='accuracy', ascending=False)

        # Using the sorted_models_df from above, we are creating another dict that has the models fitted with the parameters.
        mlflow_models = {}
        for key, value in models.items():
            if key in sorted_models_df.index:
                if key == 'Bagging_Classifier':
                    sorted_models_df.params['Bagging_Classifier'] = {'n_estimators': sorted_models_df.params['Bagging_Classifier']['n_estimators'],
                                                                     'n_jobs': sorted_models_df.params['Bagging_Classifier']['n_jobs'],
                                                                     'estimator': sorted_models_df.params['Bagging_Classifier']['estimator']}
                mlflow_models[key] = value(**(sorted_models_df.params[key]))

        # Create the list[tuple] best_estimators to fit in the voting classifier
        best_estimators_mlflow = list(
            zip(mlflow_models.keys(), mlflow_models.values()))

        # If using stacking classifier, get the final estimator using:
        final_estimator_mlflow = {key: value(**(sorted_models_df.iloc[:1, :].params[key]))
                                  for key, value in models.items() if key in sorted_models_df.iloc[:1, :].index}
        # Access the final estimator model using:

        final_estimator = final_estimator_mlflow[sorted_models_df.iloc[:1, :].index.values[0]]

        return best_estimators_mlflow, final_estimator_mlflow, final_estimator


def stacking_clf_trainer(best_estimators: list[tuple], final_estimator,
                         x_train: pd.DataFrame, y_train: pd.DataFrame,
                         x_test: pd.DataFrame, y_test: pd.DataFrame,
                         client: MlflowClient,
                         model_path: str):
    stacking_clf = StackingClassifier(estimators=best_estimators,
                                      final_estimator=final_estimator,
                                      cv=3,
                                      n_jobs=-1,
                                      verbose=3,
                                      passthrough=True)

# Test with Stacking_classifier
    y_pred_stacking_clf = model_trainer_2(x_train=x_train, y_train=y_train,
                                          x_test=x_test, y_test=y_test,
                                          model=stacking_clf)
    metrics_stacking_clf = eval_metrics(
        y_true=y_test, y_pred=y_pred_stacking_clf)
    metrics_stacking_clf['model'] = stacking_clf
    metrics_stacking_clf['model_name'] = 'Stacked_Classifier'

# Log Stacking_CLF in MLFlow
    tags = {
        "metrics": "['Balanced_Accuracy_Score', 'F1_Score', 'Accuracy_Score', 'Cost']"}
    exp_id_stack_clf = mlflow.create_experiment(
        name=f"{exp_count}_Stacked_Classifier_{exp_count}", tags=tags)
    with mlflow.start_run(experiment_id=exp_id_stack_clf,
                          run_name=f"Challenger {stacking_clf.__class__.__name__}",
                          tags={'model': 'Stacked_Classifier',
                                "run_type": "parent",
                                "model_type": "Challenger"}):
        # parent_run_id = parent_run.info.run_id
        mlflow_logger(model=stacking_clf,
                      client=client,
                      model_name='Stacked_Classifier',
                      should_log_parent_model=False,
                      artifact_path='candidate_Stacked_Classifier')

        mlflow.log_metrics(
            metrics={"Accuracy_Score": metrics_stacking_clf['Accuracy_Score']})
        tags = {'model_type': "Challenger"}
        mlflow_logger(client=client,
                      model_name='Stacked_Classifier',
                      exp_id=exp_id_stack_clf,
                      registered_model_name="Challenger Stacked_Classifier",
                      artifact_path=None,
                      tags=tags)
    save_binary(file=stacking_clf, filepath=model_path)
    return (metrics_stacking_clf, exp_id_stack_clf, y_pred_stacking_clf)

    # models['Stacked_Classifier']=StackingClassifier(**report['Stacked_Classifier']['Fittable_Params'])


def voting_clf_trainer(best_estimators: list[tuple],
                       x_train: pd.DataFrame, y_train: pd.DataFrame,
                       x_test: pd.DataFrame, y_test: pd.DataFrame,
                       client: MlflowClient,
                       model_path: Path):
    voting_clf = VotingClassifier(estimators=best_estimators,
                                  n_jobs=-1,
                                  verbose=True)
# Test with Voting_Classifier
    y_pred_voting_clf = model_trainer_2(x_train=x_train, y_train=y_train,
                                        x_test=x_test, y_test=y_test,
                                        model=voting_clf)
    metrics_voting_clf = eval_metrics(y_true=y_test, y_pred=y_pred_voting_clf)
    metrics_voting_clf['model'] = voting_clf
    metrics_voting_clf['model_name'] = 'Voting_Classifier'

# Log Voting_CLF in MLFlow
    tags = {
        "metrics": "['Balanced_Accuracy_Score', 'F1_Score', 'Accuracy_Score', 'Cost']"}
    exp_id_voting_clf = mlflow.create_experiment(name=f"{exp_count}_Voting_Classifier_{exp_count}", tags=tags)
    with mlflow.start_run(experiment_id=exp_id_voting_clf,
                          run_name=f"Challenger {voting_clf.__class__.__name__}",
                          tags={'model': 'Voting_Classifier',
                                "run_type": "parent",
                                "model_type": "Challenger"}):
        # parent_run_id = parent_run.info.run_id
        mlflow_logger(model=voting_clf,
                      client=client,
                      model_name='Voting_Classifier',
                      should_log_parent_model=False,
                      artifact_path='candidate_Voting_Classifier')

        mlflow.log_metrics(
            metrics={"Accuracy_Score": metrics_voting_clf['Accuracy_Score']})
        tags = {'model_type': "Challenger"}
        mlflow_logger(client=client,
                      model_name='Voting_Classifier',
                      exp_id=exp_id_voting_clf,
                      registered_model_name="Challenger Voting_Classifier",
                      artifact_path=None,
                      tags=tags)
    save_binary(file=voting_clf, filepath=model_path)
    return (metrics_voting_clf, exp_id_voting_clf, y_pred_voting_clf)


def model_trainer(mlflow_experiment_id, client: MlflowClient,
                  x_train: pd.DataFrame, y_train: pd.DataFrame,
                  x_test: pd.DataFrame, y_test: pd.DataFrame,
                  model_path: Path,
                  model=None, model_dict: dict = None,
                  params: dict = None):
    y_pred_final_estimator = model_trainer_2(x_train=x_train, y_train=y_train,
                                             x_test=x_test, y_test=y_test,
                                             model=model)

    metrics_final_estimator = eval_metrics(
        y_true=y_test, y_pred=y_pred_final_estimator)
    metrics_final_estimator['model'] = model
    metrics_final_estimator['model_name'] = list(model_dict.keys())[0]
    run_id = 0

    with mlflow.start_run(experiment_id=mlflow_experiment_id,
                          run_name=f"Challenger {metrics_final_estimator['model'].__class__.__name__}",
                          tags={
                              'model': f"{metrics_final_estimator['model'].__class__.__name__}",
                              "run_type": "parent",
                              "model_type": "Challenger"}
                          ) as final_estimator_run:

        mlflow.sklearn.log_model(sk_model=metrics_final_estimator['model'],
                                 artifact_path=f"challenger_{metrics_final_estimator['model'].__class__.__name__}")

        run_id = final_estimator_run.info.run_id
        tags = {'model_type': "Challenger"}
        artifact_path = client.list_artifacts(run_id=run_id)[0].path
        print(f"\n\nArtifact Path: {artifact_path}\n\n")
        artifact_uri = client.get_run(run_id).info.artifact_uri
        source = artifact_uri+'/'+artifact_path

        client.create_registered_model(name=f"Challenger {metrics_final_estimator['model'].__class__.__name__}",
                                       description="This model has been trained on the entire training dataset",
                                       tags=tags)
        client.create_model_version(name=f"Challenger {metrics_final_estimator['model'].__class__.__name__}",
                                    source=source,
                                    run_id=run_id)

        mlflow.log_metrics(eval_metrics(
            y_true=y_test, y_pred=y_pred_final_estimator))
    save_binary(file=model, filepath=model_path)
    return metrics_final_estimator, y_pred_final_estimator


def model_trainer_2(model, x_train: pd.DataFrame, y_train: pd.DataFrame,
                    x_test: pd.DataFrame, y_test: pd.DataFrame,
                    params: dict = None):
    if params:
        model.set_params(**params)
        model.fit(X=x_train, y=y_train)
    else:
        model.fit(X=x_train, y=y_train)
    return (model.predict(X=x_test))


def mlflow_logger(artifact_path: str, client: MlflowClient, metrics_: dict = None, data=None,
                  model=None, model_name: str = None, is_tuning_complete: bool = False,
                  should_log_parent_model: bool = False, should_register_model: bool = False,
                  registered_model_name: str = None, tags: dict = None,
                  run_id: str = None, exp_id: int | list = None):
    if should_register_model is False and not artifact_path:
        # x_train=data
        print("Registering Best Parent Model")
        print("Client_Tracking_URI: ", client.tracking_uri)
        print("Client_Registry_URI: ", client._registry_uri)
        filter_string = "tags.run_type ilike 'parent'"
        best_run_id = mlflow.search_runs(experiment_ids=[exp_id],
                                         order_by=['metrics.Accuracy_Score DESC'],
                                         filter_string=filter_string)[['run_id', 'artifact_uri', 'metrics.Accuracy_Score']]['run_id'][0]
        best_artifact_path = mlflow.search_runs(experiment_ids=[exp_id],
                                                order_by=[
                                                    'metrics.Accuracy_Score DESC'],
                                                filter_string=filter_string)[['run_id', 'artifact_uri', 'metrics.Accuracy_Score']]['artifact_uri'][0]
        artifact_path_name = client.list_artifacts(f'{best_run_id}')[0].path
        print(f"\nBest_Run_ID: {best_run_id}")
        print(
            f"Best_Model's_Artifact_Path: {best_artifact_path}/{artifact_path_name}")

        if registered_model_name == 'Challenger Voting_Classifier' or registered_model_name == 'Challenger Stacked_Classifier':
            description = "This model has been trained on the entire training dataset"
        else:
            description = None

        client.create_registered_model(
            name=registered_model_name, description=description, tags=tags)
        client.create_model_version(name=registered_model_name,
                                    source=f"{best_artifact_path}/{artifact_path_name}",
                                    run_id=best_run_id)

    elif not artifact_path and should_register_model is True:
        # print(f"Registering Best Model so far as {registered_model_name}")
        parent_runs = mlflow.search_registered_models()
        print("Experiment IDs: ", exp_id)
        if is_tuning_complete is False:
            runs_df = mlflow.search_runs(experiment_ids=exp_id,
                                         search_all_experiments=True,
                                         filter_string="tags.run_type ilike 'parent'")
        else:
            runs_df = mlflow.search_runs(experiment_ids=exp_id,
                                         search_all_experiments=True,
                                         filter_string="tags.run_type ilike 'parent' and tags.model_type ilike 'Challenger'")
        runs_list_ = [
            parent_runs[i].latest_versions[0].run_id for i in range(len(parent_runs))]
        best_run = runs_df[runs_df['run_id'].isin(runs_list_)].sort_values(
            by="metrics.Accuracy_Score", ascending=False).reset_index(drop=True)['run_id'][0]
        best_artifact = runs_df[runs_df['run_id'].isin(runs_list_)].sort_values(
            by="metrics.Accuracy_Score", ascending=False).reset_index(drop=True)['artifact_uri'][0]
        artifact_path_name = client.list_artifacts(f'{best_run}')[0].path
        best_exp_id = runs_df[runs_df['run_id'] ==
                              best_run]['experiment_id'].values[0]
        model_name = json.loads(runs_df[runs_df['run_id'].isin(runs_list_)].sort_values(
            by="metrics.Accuracy_Score", ascending=False).reset_index(drop=True)['tags.mlflow.log-model.history'][0])[0]['artifact_path']
        # model_name=model_name.replace("Optuna for ", "")

        # if is_tuning_complete == False:
        #     client.set_tag(best_run, "model_type", "Challenger")
        # if registered_model_name == "Best HP Tuned Candidate":
        #     description="This model has been trained on batches"
        # else:
        #     description=f"{model_name} is a {registered_model_name} model"
        if is_tuning_complete is True:
            model_name = re.sub(r'Challenger ', '', model_name)
            client.create_registered_model(name=f"{registered_model_name} {model_name}",
                                           tags={
                                               "model_type": f"{registered_model_name}"},
                                           description=description)

            client.create_model_version(name=f"{registered_model_name} {model_name}",
                                        source=f"{best_artifact}/{artifact_path_name}",
                                        run_id=best_run,
                                        tags={"model_type": f"{registered_model_name}",
                                              "model_name": model_name})
        else:
            model_name = re.sub(r'candidate_', '', model_name)
            client.set_registered_model_tag(name=f"{registered_model_name}{model_name}",
                                            key='model_type',
                                            value='Best_HP_Tuned_Candidate')
        return (best_exp_id)

    elif should_log_parent_model is True and should_register_model is False:
        print("Logging Parent Model")
        # x_train, x_test, y_train, y_test=data
        print("Experiment IDs: ", exp_id)
        # print("Tracking URI: ",mlflow.get_tracking_uri())
        # print("Registry URI: ",client._registry_uri)
        filter_string = f"tags.mlflow.parentRunId ILIKE '{run_id}'"
        best_run_id = mlflow.search_runs(experiment_ids=[exp_id],
                                         filter_string=filter_string,
                                         order_by=['metrics.Accuracy_Score DESC'])[['run_id', 'artifact_uri', 'metrics.Accuracy_Score']]['run_id'][0]
        best_artifact_path = mlflow.search_runs(experiment_ids=[exp_id],
                                                filter_string=filter_string,
                                                order_by=['metrics.Accuracy_Score DESC'])[['run_id', 'artifact_uri', 'metrics.Accuracy_Score']]['artifact_uri'][0]
        artifact_path_name = client.list_artifacts(f'{best_run_id}')[0].path
        print(f"Parent_Run_ID: {run_id}")
        print(f"Artifact_Path: {best_artifact_path}/{artifact_path_name}")
        if model_name == 'XGB_Classifier':
            best_model = mlflow.xgboost.load_model(
                f"{best_artifact_path}/{artifact_path_name}")
            params = client.get_run(best_run_id).data.params
            for key, value in params.items():
                try:
                    params[key] = eval(value)
                except Exception:
                    params[key] = value
                    if value == 'nan':
                        params[key] = np.nan
            print("Best Params:\n", {key: value for key, value in params.items() if value is not None}, "\n")
            # signature=mlflow.xgboost.infer_signature(model_input=x_train,
            #                                             model_output=best_model.predict(x_train),
            #                                             params={key: value for key, value in params.items() if value is not None})
            mlflow.xgboost.log_model(xgb_model=best_model,
                                     artifact_path=artifact_path)
            # signature=signature)
        elif model_name == 'Light_GBM':
            best_model = mlflow.lightgbm.load_model(
                f"{best_artifact_path}/{artifact_path_name}")
            mlflow.lightgbm.log_model(lgb_model=model,
                                      artifact_path=artifact_path)
            params = client.get_run(best_run_id).data.params
        else:
            # print("Tracking URI: ",client.tracking_uri)
            # print("Registry URI: ",client._registry_uri)
            best_model = mlflow.sklearn.load_model(
                f"{best_artifact_path}/{artifact_path_name}")
            params = client.get_run(best_run_id).data.params
            if model_name == "Stacked_Classifier" or model_name == "Voting_Classifier":
                # signature=mlflow.models.infer_signature(model_input=x_train,
                #                                         model_output=best_model.predict(x_train),
                #                                         params=params)
                mlflow.sklearn.log_model(sk_model=best_model,
                                         artifact_path=artifact_path)
                # signature=signature)
                for key, value in params.items():
                    try:
                        params[key] = eval(value)
                    except Exception:
                        params[key] = value
                        if value == 'nan':
                            params[key] = np.nan
                params = {key: value for key,
                          value in params.items() if value is not None}
            else:
                for key, value in params.items():
                    try:
                        params[key] = eval(value)
                    except Exception:
                        params[key] = value
                        if value == 'nan':
                            params[key] = np.nan
                # signature=mlflow.models.infer_signature(model_input=x_train,
                #                                             model_output=best_model.predict(x_train),
                #                                             params={key: value for key, value in params.items() if value is not None})
                mlflow.sklearn.log_model(sk_model=best_model,
                                         artifact_path=artifact_path)
                # signature=signature)
        print("Best Params:\n", {
              key: value for key, value in params.items() if value is not None}, "\n")
        if model_name == "Stacked_Classifier" or model_name == 'Voting_Classifier':
            flag = 0
            for i in range(len(best_model.get_params()['estimators'])):
                if 'XGB_Classifier' == best_model.get_params()['estimators'][i][0]:
                    flag = 1
            if flag == 1:
                clf_list = []
                for i in range(len(best_model.get_params()['estimators'])):
                    # if best_model.get_params()['estimators'][i][0] != "XGB_Classifier":
                    clf_list.append((best_model.get_params()[
                                    'estimators'][i][0], f"{best_model.get_params()['estimators'][i][1].__class__.__name__}()"))

                # estimators_params={}
                # for i in range(len(best_model.get_params()['estimators'])):
                #     estimators_params[best_model.get_params()['estimators'][i][0]]=best_model.get_params()['estimators'][i][1].get_params()
                # for i in estimators_params:
                #     estimators_params[i]={key:value for key,value in estimators_params[i].items() if value is not None}

                if model_name == "Stacked_Classifier":
                    s_clf_params = {'estimators': clf_list,
                                    'final_estimator': f'{best_model.get_params()["final_estimator"].__class__.__name__}()',
                                    'cv': best_model.get_params()['cv'],
                                    'stack_method': best_model.get_params()['stack_method'],
                                    'passthrough': best_model.get_params()['passthrough']}
                    # for key,value in estimators_params.items():
                    #     s_clf_params[f"{key}_Params"]=value

                    print("Processed_NEW_S_CLF_Params: ", s_clf_params)
                    mlflow.log_params(params=s_clf_params)

                elif model_name == "Voting_Classifier":
                    v_clf_params = {'estimators': clf_list}
                    # for key,value in estimators_params.items():
                    #     v_clf_params[f"{key}_Params"]=value
                    print("Processed_NEW_V_CLF_Params: ", v_clf_params)
                    mlflow.log_params(params=v_clf_params)

            else:
                clf_list = []
                for i in range(len(best_model.get_params()['estimators'])):
                    clf_list.append((best_model.get_params()[
                                    'estimators'][i][0], best_model.get_params()['estimators'][i][1].__class__()))

                # estimators_params={}
                # for i in range(len(best_model.get_params()['estimators'])):
                #         estimators_params[best_model.get_params()['estimators'][i][0]]=best_model.get_params()['estimators'][i][1].get_params()
                # for i in estimators_params:
                #         estimators_params[i]={key:value for key,value in estimators_params[i].items() if value is not None}

                if model_name == "Stacked_Classifier":
                    s_clf_params = {'estimators': clf_list,
                                    'final_estimator': f'{best_model.get_params()["final_estimator"].__class__.__name__}()',
                                    'cv': best_model.get_params()['cv'],
                                    'stack_method': best_model.get_params()['stack_method'],
                                    'passthrough': best_model.get_params()['passthrough']}
                    # for key,value in estimators_params.items():
                    #         s_clf_params[f"{key}_params"]=value
                    print("Processed_NEW_S_CLF_Params: ", s_clf_params)
                    mlflow.log_params(params=s_clf_params)

                elif model_name == "Voting_Classifier":
                    v_clf_params = {'estimators': clf_list}
                    # for key,value in estimators_params.items():
                    #     v_clf_params[f"{key}_Params"]=value
                    print("Processed_NEW_V_CLF_Params: ", v_clf_params)
                    mlflow.log_params(params=v_clf_params)

                # new_params=best_model.get_params()
                # for key,value in new_params.items():
                #     try:
                #         new_params[key]=eval(value)
                #     except:
                #         new_params[key]=str(value)
                #         if value == 'nan':
                #             new_params[key]=np.nan
                # processed_new_params={key: value for key, value in new_params.items() if value is not None}
                # print("Processed_NEW_Params: ",processed_new_params)
                # mlflow.log_params(params=processed_new_params)
        else:
            if model_name == 'XGB_Classifier':
                params = client.get_run(best_run_id).data.params
                mlflow.log_params(params=params)
            else:
                mlflow.log_params(params=best_model.get_params())

        if model_name == "Stacked_Classifier" or model_name == 'Voting_Classifier':
            # mlflow.log_metrics(metrics=eval_metrics(y_test , best_model.fit(x_train, y_train).predict(x_test)))
            mlflow.log_metrics(metrics=metrics_)
        else:
            # mlflow.log_metrics(metrics=eval_metrics(y_test , best_model.set_params(**params).fit(x_train, y_train).predict(x_test)))
            mlflow.log_metrics(metrics=metrics_)

    else:
        print("Logging params and child model")
        # x_train, y_test, y_pred=data
        # mlflow.log_metrics(metrics=eval_metrics(y_test , y_pred))
        if model_name == "Stacked_Classifier" or model_name == 'Voting_Classifier':
            flag = 0
            for i in range(len(model.get_params()['estimators'])):
                if 'XGB_Classifier' == model.get_params()['estimators'][i][0]:
                    flag = 1
            if flag == 1:
                clf_list = []
                for i in range(len(model.get_params()['estimators'])):
                    # if model.get_params()['estimators'][i][0] != "XGB_Classifier":
                    clf_list.append((model.get_params()[
                                    'estimators'][i][0], f"{model.get_params()['estimators'][i][1].__class__.__name__}()"))

                # estimators_params={}
                # for i in range(len(model.get_params()['estimators'])):
                #     estimators_params[model.get_params()['estimators'][i][0]]=model.get_params()['estimators'][i][1].get_params()
                # for i in estimators_params:
                #     estimators_params[i]={key:value for key,value in estimators_params[i].items() if value is not None}

                if model_name == "Stacked_Classifier":
                    s_clf_params = {'estimators': clf_list,
                                    'final_estimator': f'{model.get_params()["final_estimator"].__class__.__name__}()',
                                    'cv': model.get_params()['cv'],
                                    'stack_method': model.get_params()['stack_method'],
                                    'passthrough': model.get_params()['passthrough']}
                    # for key,value in estimators_params.items():
                    #     s_clf_params[f"{key}_Params"]=value

                    print("Processed_NEW_S_CLF_Params: ", s_clf_params)
                    mlflow.log_params(params=s_clf_params)

                elif model_name == "Voting_Classifier":
                    v_clf_params = {'estimators': clf_list}
                    # for key,value in estimators_params.items():
                    #     v_clf_params[f"{key}_Params"]=value
                    print("Processed_NEW_V_CLF_Params: ", v_clf_params)
                    mlflow.log_params(params=v_clf_params)

            else:
                clf_list = []
                for i in range(len(model.get_params()['estimators'])):
                    clf_list.append((model.get_params()['estimators'][i][0], model.get_params()[
                                    'estimators'][i][1].__class__()))

                # estimators_params={}
                # for i in range(len(model.get_params()['estimators'])):
                #         estimators_params[model.get_params()['estimators'][i][0]]=model.get_params()['estimators'][i][1].get_params()
                # for i in estimators_params:
                #         estimators_params[i]={key:value for key,value in estimators_params[i].items() if value is not None}

                if model_name == "Stacked_Classifier":
                    s_clf_params = {'estimators': clf_list,
                                    'final_estimator': f'{model.get_params()["final_estimator"].__class__.__name__}()',
                                    'cv': model.get_params()['cv'],
                                    'stack_method': model.get_params()['stack_method'],
                                    'passthrough': model.get_params()['passthrough']}
                    # for key,value in estimators_params.items():
                    #         s_clf_params[f"{key}_params"]=value
                    print("Processed_NEW_S_CLF_Params: ", s_clf_params)
                    mlflow.log_params(params=s_clf_params)

                elif model_name == "Voting_Classifier":
                    v_clf_params = {'estimators': clf_list}
                    # for key,value in estimators_params.items():
                    #     v_clf_params[f"{key}_Params"]=value
                    print("Processed_NEW_V_CLF_Params: ", v_clf_params)
                    mlflow.log_params(params=v_clf_params)

        else:
            mlflow.log_params(params=model.get_params())
        if model_name == 'XGB_Classifier':
            # signature=mlflow.xgboost.infer_signature(model_input=x_train,
            #                                           model_output=model.predict(x_train),
            #                                           params={key: value for key, value in model.get_params().items() if value is not None})
            mlflow.xgboost.log_model(xgb_model=model,
                                     artifact_path=artifact_path)
            # signature=signature)

        elif model_name == 'Light_GBM':
            mlflow.lightgbm.log_model(lgb_model=model,
                                      artifact_path=artifact_path)

        elif model_name == "Stacked_Classifier" or model_name == 'Voting_Classifier':
            params = model.get_params()
            for key, value in params.items():
                params[key] = str(value)
            # signature=mlflow.models.infer_signature(model_input=x_train,
            #                                           model_output=model.predict(x_train),
            #                                           params=params)
            mlflow.sklearn.log_model(sk_model=model,
                                     artifact_path=artifact_path)
            # signature=signature)

        else:
            # signature=mlflow.models.infer_signature(model_input=x_train,
            #                                           model_output=model.predict(x_train),
            #                                           params={key: value for key, value in model.get_params().items() if value is not None})
            mlflow.sklearn.log_model(sk_model=model,
                                     artifact_path=artifact_path)
            # signature=signature)


def batch_data_create(data: pd.DataFrame):
    from sklearn.utils import shuffle

    batch_size = 5000
    num_batches = data.shape[0] // batch_size
    data_shuffled = shuffle(data, random_state=42)

    batch_list = []
    # Iterate through batches
    for i in range(num_batches):
        # Extract a batch
        start_idx = i * batch_size
        end_idx = ((i + 1) * batch_size)
        # print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        batch_list.append(data_shuffled.iloc[start_idx:end_idx, :])

        # To print the details of batch_data
        sum1 = 0
        sum0 = 0
        for i in range(len(batch_list)):
            print(f"Batch: {i}")
            sum0 += batch_list[i]['class'].value_counts()[0]
            sum1 += batch_list[i]['class'].value_counts()[1]
            print(batch_list[i]['class'].value_counts(), '\n')

        print(f"Total no of Class 0 in all batches: {sum0}")
        print(f"Total no of Class 1 in all batches: {sum1}")

    return batch_list


# DEPRECATED
def params_extractor(data: pd.DataFrame):

    params_yaml = load_yaml(PARAMS_PATH)

    data = data[data['state'] == 'COMPLETE']

    data.sort_values(by='value', ascending=False, inplace=True)

    params_name = data.columns[(data.columns.str.contains('params')) | (
        data.columns.str.contains('Model_name')) | (data.columns.str.contains('value'))]
    params = {}
    for i in data[params_name]['Model_name']:
        params[i] = {}
        for j in params_yaml['optuna'][i].keys():
            params[i][j] = data[params_name][data[params_name]
                                             ['Model_name'] == i][f'params_{j}'].values.tolist()
            params[i]['accuracy_value'] = data[params_name][data[params_name]
                                                            ['Model_name'] == i]['value'].values.tolist()

    return (params)
# DEPRECATED


def params_evaluator(sample_params: dict):
    # sample_params=client.get_run('cd460708497e4677a2cdcadeaefd8878').data.params
    for key, value in sample_params.items():
        try:
            sample_params[key] = eval(value)
        except Exception:
            sample_params[key] = value
            if value == 'nan':
                sample_params[key] = np.nan
    return sample_params
