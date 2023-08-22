from logging import Logger
from SFXGBoost.config import Config, MyLogger
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from SFXGBoost.Model import SFXGBoost
from sklearn.model_selection import train_test_split

def split_shadow(D_Shadow):
    """Splits the shadow_dataset such that it can be used for training with D_Train_Shadow and D_Out_Shadow. 

    Args:
        D_Shadow Tuple(nd.array): a Tuple with two arrays for X and y. y is One Hot encoded. 

    Returns:
        Tuple(nd.array), Tuple(ndarray): D_Train_Shadow, D_Out_Shadow
    """
    X = D_Shadow[0]
    y = D_Shadow[1]

    split = len(X[:, 0]) // 2 # find what one third of the users are
    D_Train_Shadow = (X[:split, :], y[:split, :])
    D_Out_Shadow = (X[split:, :], y[split:, :])

    return D_Train_Shadow, D_Out_Shadow

def federated_split(D_Shadow):
    num_shadows = len(D_Shadow)
    D_Shadow[0:num_shadows-1], D_Shadow[num_shadows]

def f_random(D_Train_Shadow, D_Out_Shadow):
    """splits D_Train_Shadow and D_Out_Shadow such that they can be used for training the attack model.
    The features should be picked randomly from either D_Train_Shadow or D_Out_Shadow, 
    the corresponding label should be 0 if from Out_Shadow and 1 if from Train_Shadow

    Args:
        D_Train_Shadow (Tuple(np.ndarray)): holds X and y
        D_Out_Shadow (Tuple(np.ndarray)): holds X and y
    """
    min_lenght = np.min((D_Train_Shadow[0].shape[0], D_Out_Shadow[0].shape[0])) # make it such that the concatenated list is 50/50 split
    X_Train_Shadow = D_Train_Shadow[0][:min_lenght, :]
    X_Out_Shadow = D_Out_Shadow[0][:min_lenght, :]

    # add an extra column with 1 if Train_Shadow else 0
    X_Train_Shadow = np.hstack( (X_Train_Shadow, np.ones((X_Train_Shadow.shape[0], 1)))) 
    X_Out_Shadow = np.hstack( (X_Out_Shadow, np.zeros((X_Out_Shadow.shape[0], 1))))

    #concatinate them
    X_Train_Attack = np.vstack((X_Train_Shadow, X_Out_Shadow))

    #shuffle them
    np.random.shuffle(X_Train_Attack)

    #remove and take labels
    labels = X_Train_Attack[:, -1]  # take last column
    X_Train_Attack = X_Train_Attack[:, :-1]  # take everything but the last column
    # print(f"labels = {labels.shape}")
    # print(f"X_Train_Attack = {X_Train_Attack.shape}")
    return X_Train_Attack, labels

def create_D_attack_centralised(shadow_model_s, D_Train_Shadow, D_Out_Shadow):
    # Shadow_model_s can be multiple shadow_models! TODO deal with that!
    x, labels = f_random(D_Train_Shadow, D_Out_Shadow)
    z = shadow_model_s.predict_proba(x)
    # z_top_indices = np.argsort(z)[::-1][:3] # take top 3 sorted by probability
    # z = np.take(z, z_top_indices) # take top 3
    return z, labels

def create_D_attack_federated(D_Train_Shadow, D_Out_Shadow, X_train, X_test, G_shadows, H_shadows, shadow_models, target_model):
    """creates the attack dataset to train and test for the federated attack

    Args:
        D_Train_Shadow (_type_): _description_
        D_Out_Shadow (_type_): _description_
        X_train (_type_): _description_
        X_test (_type_): _description_
        G_shadows (_type_): _description_
        H_shadows (_type_): _description_
        shadow_models (_type_): _description_

    Returns:
        _type_: _description_
    """
    # D_train_attack should be D_train_shadow label = 1, D_Out_Shadow = 0
    for i in range(len(D_Train_Shadow)):
        x, labels = f_random(D_Train_Shadow[i], D_Out_Shadow)
        z = shadow_models[i].predict_proba(x)
        for depth in maxDepth:
            G = G_shadows[depth][i]
            H = H_shadows[depth][i]
    target_model.re
    return D_train_Attack, D_test_Attack

def preform_attack_centralised(config:Config, logger:Logger, D_Shadow, target_model, shadow_model, attack_model, X, y, X_test, y_test, fName=None) -> np.ndarray:
    """ Depricated, tmp

    Args:
        config (Config, D_Shadow, target_model, shadow_model, attack_model, X, y, fName, optional): _description_. Defaults to None)->np.array(.
    """
    D_Train_Shadow, D_Out_Shadow = split_shadow(D_Shadow)
    
    y_pred=None
    if type(shadow_model) != SFXGBoost:
        y_pred = shadow_model.fit(D_Train_Shadow[0], np.argmax(D_Train_Shadow[1], axis=1)).predict(X_test)
    else:
        y_pred = shadow_model.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName).predict(X_test)
    
    Metric_shadow_acc = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    del y_pred
    
    x, label = f_random(D_Train_Shadow, D_Out_Shadow)
    
    X_train, X_test, label_train, label_test = train_test_split(x, label, test_size=0.2, random_state=12)
    z_train = shadow_model.predict_proba(X_train)
    z_test = target_model.predict_proba(X) # todo test data outside
    test_x, test_label = f_random((X,y), (X_test, y_test))
    
    
    attack_model.fit(z_train, label_train)
    
    y_pred = attack_model.predict(target_model.predict_proba(test_x))
    Metric_attack_acc = accuracy_score(test_label, y_pred)
    logger.warning(f"DEBUG: accuracy attack: {Metric_attack_acc}")
    print(f"DEBUG: accuracy attack: {Metric_attack_acc}")
    Metric_attack_precision = precision_score(test_label, y_pred)
    print(f"DEBUG: precision attack: {Metric_attack_precision}")
    logger.warning(f"DEBUG: precision attack: {Metric_attack_precision}")
    print(f"DEBUG: {y_pred}")
    logger.warning("DEBUG: y_pred = {y_pred}")
    logger.warning(f"DEBUG: f(x) target = {target_model.predict_proba(test_x)}")




