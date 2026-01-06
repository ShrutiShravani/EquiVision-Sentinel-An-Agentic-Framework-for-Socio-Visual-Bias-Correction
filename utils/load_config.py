import yaml
import os

def load_config(config_path="config\path.yaml"):
    """ load config path"""

    with open(config_path,'r') as file:
        config=yaml.safe_load(file)
    return config
