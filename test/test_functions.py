from box.config_box import ConfigBox
import pytest
from src.utils import (dynamic_threshold, load_yaml,
                       schema_saver, train_test_splitter, eval_metrics)
import numpy as np


def test_dynamic_threshold(data_dynamic_threshold):
    threshold = dynamic_threshold(data_dynamic_threshold)
    assert threshold == np.quantile(data_dynamic_threshold[:-1], 0.25)


@pytest.mark.parametrize('data_load_yaml', ['params.yaml', None])
def test_load_yaml(data_load_yaml):
    if data_load_yaml == 'params.yaml':
        assert type(load_yaml(data_load_yaml)) == ConfigBox
    else:
        with pytest.raises(Exception):
            _ = load_yaml(data_load_yaml)


def test_stage_1_processing(data_stage_1):
    dataframe, schema = data_stage_1
    assert "ident_id" not in dataframe.columns
    assert 'class' in dataframe.columns
    assert all(dataframe['class'].isin([1, 0]))
    assert (dataframe != 'na').any().any()
    assert list(dataframe.drop(columns='class').columns) == schema
    assert dataframe[schema].dtypes.unique() == float


def test_stage_2_processing(data_stage_2):
    assert not data_stage_2.isna().any().any()
    assert data_stage_2['class'].value_counts()[0] == data_stage_2['class'].value_counts()[1]


def test_train_test_splitting(data_stage_2):
    train_df, test_df = train_test_splitter(data_stage_2)
    assert test_df.shape[0] == (data_stage_2.shape[0] - train_df.shape[0])


def test_schema_saver(schema_saver_fixture):
    filepath, test_data = schema_saver_fixture
    schema_saver(dataframe=test_data, filepath=filepath)
    test_schema = load_yaml(filepath)
    assert list(test_schema['Features'].keys()) == list(test_data.drop(columns='class').columns)
    assert list(test_schema['Target'].keys()) == list(test_data[['class']].columns)


def test_eval_metrics(eval_metrics_tester):
    y_test_random, y_pred_random = eval_metrics_tester
    eval_dict = eval_metrics(y_test_random, y_pred_random)
    balanced_accuracy_score = eval_dict['Balanced_Accuracy_Score']
    F1_Score = eval_dict['F1_Score']
    Accuracy_Score = eval_dict['Accuracy_Score']
    Cost = eval_dict['Cost']
    assert all(key in eval_dict.keys() for key in ['Balanced_Accuracy_Score', 'F1_Score', 'Accuracy_Score', 'Cost'])
    assert balanced_accuracy_score <= 1
    assert F1_Score <= 1
    assert Accuracy_Score <= 1
    assert Cost > 0
