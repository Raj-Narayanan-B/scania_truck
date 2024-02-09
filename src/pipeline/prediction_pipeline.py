import pandas as pd
import numpy as np
import mlflow
from typing import Union  # type: ignore
from src.utils import load_binary  # , eval_metrics
from src.config.configuration_manager import ConfigurationManager
from src import logger


class Prediction_Pipeline(ConfigurationManager):
    def __init__(self, data: Union[pd.DataFrame, dict]):
        super().__init__()
        self.preprocessor_config = self.get_preprocessor_config()
        # self.batch_prediction_ = batch_prediction
        self.data_ = data

    def prediction_pipeline(self):
        logger.info("Loading the saved pipeline")
        preprocessor = load_binary(filepath=self.preprocessor_config.preprocessor_path)

        logger.info("Loading champion model source from MLflow")
        model_source = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Champion'")[0].latest_versions[0].source

        logger.info("Loading the Champion Model")
        model = mlflow.sklearn.load_model(model_source)

        # Batch Prediction
        if isinstance(self.data_, pd.DataFrame):
            logger.info("Commencing batch prediction")
            X = self.data_

        # Online Prediction
        elif isinstance(self.data_, dict):
            logger.info("Commencing online prediction")
            X = np.array(list(self.data_.values())).reshape(1, -1)

        logger.info("Commencing data transformation")
        X_transformed = preprocessor.transform(X)
        logger.info("Data transformation complete")

        logger.info("Commencing prediction")
        y_pred_ = model.predict(X_transformed)
        logger.info("Prediction complete")

        return y_pred_
