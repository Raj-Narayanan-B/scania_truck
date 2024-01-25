# import os
# import yaml
# # os.system("pip install -r requirements.txt")
# # os.system('dvc pull')
# # print (obj.config_path)
# # print (os.getcwd())
import mlflow


# # from src.config.configuration_manager import ConfigurationManager

# # obj = ConfigurationManager()
# print(mlflow.get_registry_uri())
# # print(mlflow.get_tracking_uri())
print(mlflow.search_runs(experiment_ids=['60']))