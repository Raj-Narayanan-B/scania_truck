import os
# import yaml
# os.system("pip install -r requirements.txt")
# os.system('dvc pull')
# print (obj.config_path)
# print (os.getcwd())
import mlflow
from src.utils import save_yaml
from pathlib import Path #type: ignore


from src.config.configuration_manager import ConfigurationManager
from src.entity.entity_config import ModelTrainerConf
obj = ConfigurationManager()
model_obj = obj.get_model_config()

# print(mlflow.get_registry_uri())
# print(mlflow.get_tracking_uri())
# print(mlflow.search_runs(experiment_ids=['60']))
print(os.getcwd())
class saver:
    def __init__(self,model_config:ModelTrainerConf):
        self.model_config = model_config
    
    def saverrr(self):
        dict = {'artifact_path_name' : 'challenger_hyperopt_SGD_Classifier'}
        save_yaml(file = dict, filepath = f"{self.model_config.root_dir}/artifact_path.yaml")


saver_obj = saver(model_config = model_obj)
saver_obj.saverrr()


