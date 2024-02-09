# from src.config.configuration_manager import ConfigurationManager
# from src.entity.entity_config import Stage1ProcessingConf, PreprocessorConf, ModelTrainerConf
# from src.utils import load_binary, eval_metrics, save_yaml
# import pandas as pd
# from imblearn.combine import SMOTETomek
# from pathlib import Path  # type: ignore


# class test_data_prediction_component:
#     def __init__(self,
#                  stage1_conf: Stage1ProcessingConf,
#                  preprocessor_conf: PreprocessorConf,
#                  model_conf: ModelTrainerConf) -> None:
#         self.stage1_config = stage1_conf
#         self.preprocessor_config = preprocessor_conf
#         self.model_config = model_conf

#     def test_data_prediction(self):
#         df = pd.read_csv(self.stage1_config.test_data_path).iloc[:1000, :]
#         X = df.drop(columns='class')
#         y = df['class']

#         preprocessor = load_binary(self.preprocessor_config.preprocessor_path)
#         model = load_binary(self.model_config.model_path)
#         smote = SMOTETomek(sampling_strategy='minority',
#                            random_state=8,
#                            n_jobs=-1)

#         X_transformed = preprocessor.transform(X)
#         X_smote, y_smote = smote.fit_resample(X=X_transformed,
#                                               y=y)

#         y_pred = model.predict(X_smote)
#         metrics = eval_metrics(y_true=y_smote,
#                                y_pred=y_pred)

#         print(metrics)
#         save_yaml(file=metrics,
#                   filepath=Path(r'artifacts\metrics\prediction_metrics.yaml'))


# config_obj = ConfigurationManager()
# stage1_obj = config_obj.get_stage1_processing_config()
# preprocessor_obj = config_obj.get_preprocessor_config()
# model_obj = config_obj.get_model_config()

# class_obj = test_data_prediction_component(stage1_conf=stage1_obj,
#                                            preprocessor_conf=preprocessor_obj,
#                                            model_conf=model_obj)
# class_obj.test_data_prediction()
