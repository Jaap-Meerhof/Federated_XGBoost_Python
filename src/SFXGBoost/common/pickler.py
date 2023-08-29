import pickle
from SFXGBoost.config import Config
import os 
from pathlib import Path
from SFXGBoost.config import Config, rank

def save(var:any, name:str, config:Config):
    file_path = Path(config.save_location)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.mkdir(parents=True, exist_ok=True)

    # if not os.path.exists(config.save_location):
    #     os.mkdir(config.save_location)
    pickle.dump(var, open(config.save_location + "/"  + name + '.p', 'wb'))

def retrieve(name:str, config:Config):
    return pickle.load(open(config.save_location + "/" + name + '.p', 'rb'))

def isSaved(name:str, config:Config):
    try:
        return os.path.exists(config.save_location + "/" + name + '.p')
    except FileNotFoundError:
        return False

def retrieveorfit(model, name, config, X, y, fName):
    if isSaved(name, config):
        model = retrieve(name, config)
        return model
    else:
        model.fit(X, y, fName)
        save(model, name=name, config=config)
        return model 