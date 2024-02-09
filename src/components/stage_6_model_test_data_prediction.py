import pandas as pd
from pprint import pprint  # type: ignore
import mlflow.sklearn
from mlflow.client import MlflowClient
import mlflow

from imblearn.combine import SMOTETomek

from src.components.stage_5_model_tuning_tracking_training import model_tuning_tracking_component
from src.utils import save_yaml, load_binary, eval_metrics, save_binary
from src import logger


class model_trainer_component(model_tuning_tracking_component):
    def __init__(self):
        super().__init__()
        self.stage_1_config = self.get_stage1_processing_config()
        self.data_split_config = self.get_data_split_config()
        self.stage_2_config = self.get_stage2_processing_config()
        self.metrics_config = self.get_metric_config()
        self.model_config = self.get_model_config()
        self.preprocessor = self.get_preprocessor_config()
        self.data_path_config = self.get_data_path_config()

    def model_training(self):
        schema = self.schema_path
        target = list(schema.Target.keys())[0]
        client = MlflowClient(tracking_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow",
                              registry_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow")

        logger.info("loading training and testing datasets")

        main_test_df = pd.read_csv(self.stage_1_config.test_data_path)

        test_data_x = main_test_df.drop(columns=target)
        test_data_y = main_test_df[target]

        logger.info("Loading saved Pipeline")
        preprocessor_pipeline = load_binary(self.preprocessor.preprocessor_path)
        logger.info("Creating SmoteTomek object")
        smote = SMOTETomek(sampling_strategy='minority', random_state=8)

        logger.info("Commencing data transformation with Pipeline and SmoteTomek")
        test_data_x_transformed = preprocessor_pipeline.transform(test_data_x)
        test_data_x_transformed_smote, test_data_y_transformed_smote = smote.fit_resample(X=test_data_x_transformed,
                                                                                          y=test_data_y)

        columns_list = list(preprocessor_pipeline.get_feature_names_out())
        X_column_names = [i for i in columns_list if i != target]
        transformed_test_df = pd.DataFrame(test_data_x_transformed_smote, columns=X_column_names)
        transformed_test_df[target] = test_data_y_transformed_smote
        transformed_test_df.to_csv(self.data_path_config.final_test_data, index=False)

        print(f"\ntransformed_test_df shape: {transformed_test_df.shape}")
        print(f"NA in transformed_test_df: {transformed_test_df.isna().sum().unique()}")
        print(f"transformed_test_df Value_Counts: {transformed_test_df[target].value_counts()}\n")

        logger.info("Transformation Complete")

        logger.info("Fetching Sources of Challenger Models from MLFlow")

        sources = []
        for i in range(3):
            sources.append(mlflow.search_registered_models(filter_string="tags.model_type ilike 'Challenger'")[i].latest_versions[0].source)

        logger.info("Loading Challenger models from MLFlow")
        logger.info("Fitting the loaded models and calculating accuracies of each model")
        report = {}
        models = {}
        for i in range(len(sources)):
            model = mlflow.sklearn.load_model(sources[i])
            model_name = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Challenger'")[i].name
            report[model_name] = eval_metrics(y_true=test_data_y_transformed_smote,
                                              y_pred=model.predict(test_data_x_transformed_smote))
            models[model_name] = model

        logger.info("Models evaluation complete")
        report_df = pd.DataFrame(report).T.sort_values(by='Accuracy_Score', ascending=False)
        report_df['Models'] = models
        print("The final report is:\n")
        pprint(report_df, compact=True)
        print("\n")

        champion_model_name = report_df.iloc[:1, :]['Accuracy_Score'].index[0]
        champion_model_accuracy_score = report_df.iloc[:1, :]['Accuracy_Score'].values[0]
        champion_model = report_df.iloc[:1, :]['Models'].values[0]

        logger.info("Champion selected")
        print(f"\nChampion Model: {champion_model_name}")
        print(f"\nChampion Model Accuracy: {champion_model_accuracy_score}")
        logger.info("Saving the champion model")
        # save_binary(file=, filepath=)

        client.set_registered_model_tag(name=champion_model_name,
                                        key='model_type',
                                        value='Champion')
        client.set_registered_model_alias(name=champion_model_name,
                                          alias='champion',
                                          version='1')

        # champion_source_ = {}
        # for i in range(1):
        #     champion_source_['Champion_Estimator'] = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Champion'")[i].latest_versions[0].source + "/model.pkl"
        # save_yaml(file=champion_source_, filepath=r"params.yaml", mode='a')

        logger.info(f"Saving best_metrics.yaml file at {self.metrics_config.best_metric}")
        save_yaml(file={"Final_Test_Data_Prediction_Report": report}, filepath=self.metrics_config.best_metric)

        logger.info("Saving prediction file as a CSV file")
        pd.DataFrame(champion_model.predict(test_data_x_transformed_smote), columns=['test_data_y_pred']).to_csv(self.data_path_config.prediction_data, index=False)

        logger.info(f"Logging Champion Model at {self.model_config.model_path}")
        save_binary(file=champion_model, filepath=self.model_config.model_path)


# obj = ConfigurationManager()
# stage_1_obj = obj.get_stage1_processing_config()
# data_split_obj = obj.get_data_split_config()
# stage_2_obj = obj.get_stage2_processing_config()
# model_metrics_obj = obj.get_metric_config()
# model_config_obj = obj.get_model_config()
# preprocessor_obj = obj.get_preprocessor_config()
# data_path_obj = obj.get_data_path_config()

# model_trainer_obj = model_trainer_component(data_split_conf=data_split_obj,
#                                             stage_1_conf=stage_1_obj,
#                                             stage_2_conf=stage_2_obj,
#                                             metrics_conf=model_metrics_obj,
#                                             model_conf=model_config_obj,
#                                             preprocessor_conf=preprocessor_obj,
#                                             data_path_conf=data_path_obj)


# model_trainer_obj.model_training()
