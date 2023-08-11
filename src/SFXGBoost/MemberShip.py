from SFXGBoost.config import Config, MyLogger
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from SFXGBoost.Model import SFXGBoost

def split_shadow(D_shadow):
    """Splits the shadow_dataset such that it can be used for training with D_Train_Shadow and D_Out_Shadow. 
    D_Test is for testing how well the shadow model performs

    Args:
        D_shadow (_type_): _description_

    Returns:
        _type_: _description_
    """
    split = len(D_shadow[0][:, 0]) // 3 # find what one third of the users are
    D_Train_Shadow = (shadow_)
    D_Out_Shadow = 
    D_Test = 
    return D_Train_Shadow, D_Out_Shadow, D_Test

def preform_attack_centralised(config:Config, D_Shadow, target_model, shadow_model, attack_model, X, y, fName=None) -> np.array():
    D_Train_Shadow, D_Out_Shadow, D_Test = split_shadow(D_shadow)
    
    y_pred=None
    if type(shadow_model) != SFXGBoost:
        y_pred = shadow_model.fit(D_Train_Shadow, D_Train_Shadow).predict(D_Test)
    else:
        y_pred = shadow_model.fit(D_Train_Shadow, D_Train_Shadow, fName).predict(D_Test)
    
    Metric_shadow_acc = accuracy_score(D_Test[1], y_pred)
    del y_pred
    
    z, label = f_random(D_Train_Shadow, D_Out_Shadow)

    attack_model.fit(z, label)
    
    y_pred = attack_model.predict(attack_test)
    Metric_attack_acc = accuracy_score(attack_test, y_pred)
    Metric_attack_precision = precision_score(attack_test, y_pred)



