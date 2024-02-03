from box.config_box import ConfigBox
import pytest
import unittest
from unittest.mock import patch, mock_open, MagicMock #type: ignore
from src.constants import SCHEMA_PATH #type: ignore
from src.utils import dynamic_threshold,load_yaml,save_yaml,stage_1_processing_function
import numpy as np

def test_dynamic_threshold(data_dynamic_threshold):
    threshold = dynamic_threshold(data_dynamic_threshold)
    assert threshold == np.quantile(data_dynamic_threshold[:-1], 0.25)

@pytest.mark.parametrize('data_load_yaml', ['params.yaml',None])
def test_load_yaml(data_load_yaml):
    if data_load_yaml == 'params.yaml':
        assert type(load_yaml(data_load_yaml)) == ConfigBox
    else:
        with pytest.raises(Exception):
            _ = load_yaml(data_load_yaml)

def test_stage_1_processing(data_stage_1):
    dataframe,schema = data_stage_1
    assert "ident_id" not in dataframe.columns
    assert 'class' in dataframe.columns
    assert all(dataframe['class'].isin([1, 0]))
    assert (dataframe != 'na').any().any()
    assert list(dataframe.drop(columns = 'class').columns) == schema
    assert dataframe[schema].dtypes.unique() == float

def test_stage_2_processing(data_stage_2):
    dataframe = data_stage_2
    assert dataframe.isna().any().any() == False
    assert dataframe['class'].value_counts()[0] == dataframe['class'].value_counts()[1]
