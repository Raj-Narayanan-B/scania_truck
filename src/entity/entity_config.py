from dataclasses import dataclass #type: ignore
from pathlib import Path #type: ignore


@dataclass(frozen=True)
class DataIngestionConf:
    train_data_1_secure_connect_bundle: Path
    train_data_1_token: Path
    train_data_1_key_space: str
    train_data_1_table: str
    train_data_1_path: Path

    train_data_2_secure_connect_bundle: Path
    train_data_2_token: Path
    train_data_2_key_space: str
    train_data_2_table: str
    train_data_2_path: Path

    train_data_3_secure_connect_bundle: Path
    train_data_3_token: Path
    train_data_3_key_space: str
    train_data_3_table: str
    train_data_3_path: Path

    test_data_1_secure_connect_bundle: Path
    test_data_1_token: Path
    test_data_1_key_space: str
    test_data_1_table: str
    test_data_1_path: Path

    test_data_2_secure_connect_bundle: Path
    test_data_2_token: Path
    test_data_2_key_space: str
    test_data_2_table: str
    test_data_2_path: Path

    test_data_3_secure_connect_bundle: Path
    test_data_3_token: Path
    test_data_3_key_space: str
    test_data_3_table: str
    test_data_3_path: Path
    
    root_directory: Path

@dataclass(frozen=True)
class DataPathConf:
    train_data1: Path
    train_data2: Path
    train_data3: Path
    test_data1: Path
    test_data2: Path
    test_data3: Path

@dataclass(frozen=True)
class Stage1ProcessingConf:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class Stage2ProcessingConf:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class DataSplitConf:
    root_dir: Path
    train_path: Path
    test_path: Path

@dataclass(frozen=True)
class ModelTrainerConf:
    root_dir: Path
    model_path: str

@dataclass(frozen=True)
class ModelMetricsConf:
    root_dir: Path
    metrics: Path
    best_metric: Path

@dataclass(frozen=True)
class PreprocessorConf:
    root_dir: Path
    preprocessor_path: str



