# import os
# import yaml
# os.system("pip install -r requirements.txt")
# os.system('dvc pull')
# print (obj.config_path)
# print (os.getcwd())
# from airflow.utils import db
# from src.utils import save_yaml
# import mlflow

# from src.config.configuration_manager import ConfigurationManager
# from src.entity.entity_config import ModelTrainerConf
# obj = ConfigurationManager()
# model_obj = obj.get_model_config()

# # print(mlflow.get_registry_uri())
# # print(mlflow.get_tracking_uri())
# # print(mlflow.search_runs(experiment_ids=['60']))
# print(os.getcwd())


# class saver:
#     def __init__(self, model_config: ModelTrainerConf):
#         self.model_config = model_config

#     def saverrr(self):
#         dict = {'artifact_path_name': 'challenger_hyperopt_SGD_Classifier'}
#         save_yaml(file=dict, filepath=f"{self.model_config.root_dir}/artifact_path.yaml")


# saver_obj = saver(model_config=model_obj)
# saver_obj.saverrr()

# champion_source = {}
# for i in range(1):
#     model_name = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Champion'")[i].name
#     model_name = model_name.replace(" ", "_")
#     champion_source[model_name] = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Champion'")[i].latest_versions[0].source + "/model.pkl"

# for i in range(2):
#     model_name = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Challenger'")[i].name
#     model_name = model_name.replace(" ", "_")
#     champion_source[model_name] = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Challenger'")[i].latest_versions[0].source + "/model.pkl"

# print(champion_source)

# save_yaml(file=champion_source, filepath='mlflow_model_sources.yaml')

# sources = {}
# for i in range(2):
#     model_name = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Challenger'")[i].name
#     if model_name != 'Challenger Stacked_Classifier' and model_name != 'Challenger Voting_Classifier':
#         model_name = 'Final_Estimator'
#     else:
#         model_name = model_name.replace(" ", "_")
#     sources[model_name] = mlflow.search_registered_models(filter_string="tags.model_type ilike 'Challenger'")[i].latest_versions[0].source + "/model.pkl"
# save_yaml(file=sources, filepath=r'artifacts\model\model_sources.yaml')


# import dvc.api
