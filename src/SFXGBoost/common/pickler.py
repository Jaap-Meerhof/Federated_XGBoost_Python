import pickle
from SFXGBoost.config import Config
import os 

def save(var:any, name:str, config:Config):
    pickle.dump(var, open(config.save_location + name + '.p', 'wb'))

def retrieve(name:str, config:Config):
    return pickle.load(open(config.save_location + name + '.p'))

def isSaved(name:str, config:Config):
    return os.path.exists(config.save_location + name + '.p')