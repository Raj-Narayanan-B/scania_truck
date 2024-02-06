import pandas as pd
from pprint import pprint  # type: ignore
import mlflow.sklearn
from mlflow.client import MlflowClient
import mlflow

from imblearn.combine import SMOTETomek

from src.config.configuration_manager import ConfigurationManager
from src.entity.entity_config import (DataSplitConf, Stage1ProcessingConf, Stage2ProcessingConf,
                                      ModelMetricsConf, ModelTrainerConf, PreprocessorConf)
from src.constants import SCHEMA_PATH
from src.utils import load_yaml, save_yaml, load_binary, eval_metrics
from src import logger
from src.components.stage_5_model_tuning_tracking_training import model_tuning_tracking_component


class model_trainer_component:
    def __init__(self,
                 data_split_conf: DataSplitConf,
                 stage_1_conf: Stage1ProcessingConf,
                 stage_2_conf: Stage2ProcessingConf,
                 metrics_conf: ModelMetricsConf,
                 model_conf: ModelTrainerConf,
                 preprocessor_conf: PreprocessorConf) -> None:
        self.stage_1_config = stage_1_conf
        self.data_split_config = data_split_conf
        self.stage_2_config = stage_2_conf
        self.metrics_config = metrics_conf
        self.model_config = model_conf
        self.preprocessor = preprocessor_conf

    def model_training(self):

        obj = model_tuning_tracking_component(stage_2_conf=self.stage_2_config,
                                              metrics_conf=self.metrics_config,
                                              model_conf=self.model_config,
                                              preprocessor_conf=self.preprocessor,
                                              data_split_conf=self.data_split_config,
                                              stage1_processor_conf=self.stage_1_config)
        obj.models_tuning()

        schema = load_yaml(SCHEMA_PATH)
        target = list(schema.Target.keys())[0]
        client = MlflowClient(tracking_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow",
                              registry_uri="https://dagshub.com/Raj-Narayanan-B/StudentMLProjectRegression.mlflow")

        # models = {'Logistic_Regression': LogisticRegression,
        #           'SGD_Classifier': SGDClassifier,
        #           'Random Forest': RandomForestClassifier,
        #           'Ada_Boost': AdaBoostClassifier,
        #           'Grad_Boost': GradientBoostingClassifier,
        #           'Bagging_Classifier': BaggingClassifier,
        #           'ExtraTreesClassifier': ExtraTreesClassifier,
        #           'Hist_Grad_Boost_Classifier': HistGradientBoostingClassifier,
        #           'Decision_Tree_Classifier': DecisionTreeClassifier,
        #           'XGB_Classifier': XGBClassifier,
        #           'Light_GBM' : LGBMClassifier,
        #           'KNN_Classifier': KNeighborsClassifier,
        #           }

        logger.info("loading training and testing datasets")

        # train_df = pd.read_csv(self.stage_1_config.train_data_path)
        main_test_df = pd.read_csv(self.stage_1_config.test_data_path)

        test_data_x = main_test_df.drop(columns=target)
        test_data_y = main_test_df[target]

        # size = None
        # stage_3_data_split_obj = data_splitting_component(data_split_conf = self.data_split_config,
        #                                                   stage1_processor_conf = self.stage_1_config)
        # train_data_training_set,train_data_testing_set = stage_3_data_split_obj.data_splitting(size)#################CHANGED

        # stage_4_final_processing_obj = stage_4_final_processing_component(data_split_conf = self.data_split_config,
        #                                                                   stage_2_processor_conf = self.stage_2_config,
        #                                                                   preprocessor_conf = self.preprocessor)
        # train_df, test_df = stage_4_final_processing_obj.final_processing(train_data_training_set,train_data_testing_set) #################CHANGED

        # x_train,y_train = train_df.drop(columns = target), train_df[target]
        # x_test,y_test = test_df.drop(columns = target), test_df[target]

        # print(f"\nx_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        # print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
        # print(f"\nNA values in x_train: {x_train.isna().sum().unique()}")
        # print(f"NA values in x_test: {x_test.isna().sum().unique()}")
        # print(f"\nTarget value counts in y_train: {y_train.value_counts()}")
        # print(f"\nTarget value counts in y_test: {y_test.value_counts()}")

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
        transformed_test_df.to_csv(r"F:\iNeuron\Projects\scania_failures_2\artifacts\data\transformed_test_df.csv", index=False)

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
        for i in range(len(sources)):
            model = mlflow.sklearn.load_model(sources[i])
            model_name = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Challenger'")[i].name
            report[model_name] = eval_metrics(y_true=test_data_y_transformed_smote,
                                              y_pred=model.predict(test_data_x_transformed_smote))

        logger.info("Models evaluation complete")
        report_df = pd.DataFrame(report).T.sort_values(by='Accuracy_Score', ascending=False)
        print("The final report is:\n")
        pprint(report_df, compact=True)

        champion_model_name = list(report_df.iloc[:1, :]['Accuracy_Score'].index)[0]
        champion_model_accuracy_score = list(report_df.iloc[:1, :]['Accuracy_Score'].values)[0]

        logger.info("Champion selected")
        print(f"\nChampion Model: {champion_model_name}")
        print(f"\nChampion Model Accuracy: {champion_model_accuracy_score}")

        client.set_registered_model_tag(name=champion_model_name,
                                        key='model_type',
                                        value='Champion')
        client.set_registered_model_alias(name=champion_model_name,
                                          alias='champion',
                                          version='1')

        # source_model_name = mlflow.search_registered_models(filter_string = f"tags.model_type ilike 'champion'")[0].latest_versions[0].name
        # source_model_name = re.sub(r"Champion ","",source_model_name)

        # if source_model_name == "Stacked_Classifier" or source_model_name == 'Voting_Classifier':
        #     print(f"Loaded model is: {source_model_name}")
        #     model = mlflow.pyfunc.load_model(model_uri = source)
        #     y_pred = model.predict(x_test)

        # elif source_model_name == 'XGB_Classifier':
        #     print(f"Loaded model is: {source_model_name}")
        #     model = mlflow.xgboost.load_model(source)
        #     model.fit(x_train,y_train)
        #     y_pred = model.predict(x_test)

        # elif source_model_name == "LGBMClassifier":
        #     print(f"Loaded model is: {source_model_name}")
        #     model = mlflow.lightgbm.load_model(source)
        #     model.fit(x_train,y_train)
        #     y_pred = model.predict(x_test)

        # else:
        #     print(f"Loaded model is: {source_model_name}")

        # model = mlflow.sklearn.load_model(source)
        # logger.info("Model loaded, now predicting the test data")
        # print(f"\nLoaded Model is: {model.__class__.__name__}")
        # # model.fit(x_train,y_train)
        # y_pred = model.predict(test_data_x_transformed_smote)

        # metrics = eval_metrics(y_true = test_data_y_transformed_smote, y_pred = y_pred)

        # print(f"\nThe final metrics from training the model is:\n{metrics}")

        logger.info(f"Saving best_metrics.yaml file at {self.metrics_config.best_metric}")
        save_yaml(file={model.__class__.__name__: report}, filepath=self.metrics_config.best_metric)


obj = ConfigurationManager()
stage_1_obj = obj.get_stage1_processing_config()
data_split_obj = obj.get_data_split_config()
stage_2_obj = obj.get_stage2_processing_config()
model_metrics_obj = obj.get_metric_config()
model_config_obj = obj.get_model_config()
preprocessor_obj = obj.get_preprocessor_config()

model_trainer_obj = model_trainer_component(data_split_conf=data_split_obj,
                                            stage_1_conf=stage_1_obj,
                                            stage_2_conf=stage_2_obj,
                                            metrics_conf=model_metrics_obj,
                                            model_conf=model_config_obj,
                                            preprocessor_conf=preprocessor_obj)


model_trainer_obj.model_training()
