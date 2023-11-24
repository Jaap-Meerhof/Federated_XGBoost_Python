import pickle
from SFXGBoost.config import Config
import os 
from pathlib import Path
from SFXGBoost.config import Config, rank

def save(var:any, name:str, config:Config=None, destination:str="Tmp/test.p"):
    file_path=None
    if config is None:
        file_path = Path(destination)
    else:
        file_path = Path(config.save_location)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.mkdir(parents=True, exist_ok=True)

    # if not os.path.exists(config.save_location):
    #     os.mkdir(config.save_location)
    if config is None:
        pickle.dump(var, open(destination, 'wb'))
    elif config.save:
        pickle.dump(var, open(config.save_location + "/" + config.dataset+ "_"  + name + '.p', 'wb'))

def retrieve(name:str, config:Config):
    return pickle.load(open(config.save_location + "/" + config.dataset+ "_" + name + '.p', 'rb'))

def isSaved(name:str, config:Config):
    try:
        return os.path.exists(config.save_location + "/" + config.dataset+ "_" + name + '.p')
    except FileNotFoundError:
        return False

def retrieveorfit(model, name, config, X, y, fName=None, X_test=None, y_test=None):
    if isSaved(name, config):
        model = retrieve(name, config)
        return model
    else:
        if fName is not None:
            if (X_test is not None) and (y_test is not None):
                model.fit(X, y, fName, X_test, y_test)
            else:
                model.fit(X, y, fName)
        else:
            model.fit(X, y)
        save(model, name=name, config=config)
        return model 