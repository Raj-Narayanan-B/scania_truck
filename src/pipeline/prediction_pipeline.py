import pandas as pd
# import numpy as np
import mlflow
from typing import Union  # type: ignore
from src.utils import eval_metrics, load_binary  # , eval_metrics
from src.config.configuration_manager import ConfigurationManager
from src.components.stage_3_data_validation import data_validation_component
from src import logger


class Prediction_Pipeline(ConfigurationManager):
    def __init__(self, data: Union[pd.DataFrame, dict]):
        super().__init__()
        self.preprocessor_config = self.get_preprocessor_config()
        # self.batch_prediction_ = batch_prediction
        self.data_ = data
        self.X = pd.DataFrame()

    def prediction_pipeline(self):
        schema = self.schema_path
        cols_to_remove = schema['columns_with_more_than_50%_missing_values']
        columns_with_0_std_dev = schema['columns_with_zero_standard_deviation']
        data_validation_obj = data_validation_component()

        logger.info("Loading the saved pipeline")
        preprocessor = load_binary(filepath=self.preprocessor_config.preprocessor_path)

        logger.info("Loading champion model source from MLflow")
        model_source = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Champion'")[0].latest_versions[0].source

        logger.info("Loading the Champion Model")
        model = mlflow.sklearn.load_model(model_source)
        logger.info(f"Model loaded is {model.__class__.__name__}")

        X = self.X
        # Batch or S3 bucket Prediction
        if isinstance(self.data_, pd.DataFrame):
            logger.info("Commencing batch prediction")
            validated_data = data_validation_obj.data_validation_(dataframe_=self.data_,
                                                                  cols_to_remove_=cols_to_remove,
                                                                  columns_with_0_std_dev_=columns_with_0_std_dev)
            X = validated_data
            if 'class' in X.columns:  # The reason for giving the following 2 lines of code under 'if' is because in real-world scenarios, 'class' may not be available.
                y = X['class']
                X.drop(columns=['class'], inplace=True)

        # Online Prediction
        elif isinstance(self.data_, dict):
            logger.info("Commencing online prediction")
            for key, value in self.data_.items():
                try:
                    self.data_[key] = eval(value)
                except Exception:
                    pass
            X_ = pd.DataFrame(self.data_, index=[0])
            X = data_validation_obj.data_validation_(dataframe_=X_,
                                                     cols_to_remove_=cols_to_remove,
                                                     columns_with_0_std_dev_=columns_with_0_std_dev)

        logger.info("Commencing data transformation")
        X_transformed = preprocessor.transform(X)
        logger.info("Data transformation complete")

        logger.info("Commencing prediction")
        y_pred_ = model.predict(X_transformed)
        logger.info("Prediction complete")

        if 'class' in X.columns:
            print(eval_metrics(y_true=y, y_pred=y_pred_))

        return y_pred_
