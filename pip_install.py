import os
import yaml
# os.system("pip install -r requirements.txt")
# os.system('dvc pull')
# print (obj.config_path)
# print (os.getcwd())

from src.config.configuration_manager import ConfigurationManager

obj = ConfigurationManager()
print(obj.config_path)