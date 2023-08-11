from SFXGBoost.config import Config, MyLogger
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from SFXGBoost.Model import SFXGBoost
from sklearn.model_selection import train_test_split

def split_shadow(D_Shadow):
    """Splits the shadow_dataset such that it can be used for training with D_Train_Shadow and D_Out_Shadow. 
    D_Test is for testing how well the shadow model performs

    Args:
        D_Shadow Tuple(nd.array): a Tuple with two arrays for X and y. y is One Hot encoded. 

    Returns:
        Tuple(nd.array), Tuple(ndarray), Tuple(ndarray): D_Train_Shadow, D_Out_Shadow, D_Test
    """
    X = D_Shadow[0]
    y = D_Shadow[1]

    split = len(X[:, 0]) // 3 # find what one third of the users are
    D_Train_Shadow = (X[:split, :], y[:split, :])
    D_Out_Shadow = (X[split:2*split, :], y[split:2*split, :])
    D_Test = (X[split*2:, :], y[split*2:, :])

    return D_Train_Shadow, D_Out_Shadow, D_Test

def f_random(D_Train_Shadow, D_Out_Shadow):
    """splits D_Train_Shadow and D_Out_Shadow such that they can be used for training the attack model.
    The features should be picked randomly from either D_Train_Shadow or D_Out_Shadow, 
    the corresponding label should be 0 if from Out_Shadow and 1 if from Train_Shadow

    Args:
        D_Train_Shadow (Tuple(np.ndarray)): holds X and y
        D_Out_Shadow (Tuple(np.ndarray)): holds X and y
    """
    max_lenght = np.max(D_Train_Shadow[0].shape[0], D_Out_Shadow[0].shape[0]) # make it such that the concatenated list is 50/50 split
    X_Train_Shadow = D_Train_Shadow[0][:max_lenght, :]
    X_Out_Shadow = D_Out_Shadow[0][:max_lenght, :]

    # add an extra column with 1 if Train_Shadow else 0
    X_Train_Shadow = np.hstack( (X_Train_Shadow, np.ones((X_Train_Shadow.shape[0], 1)))) 
    X_Out_Shadow = np.hstack( (X_Out_Shadow, np.zeros((X_Out_Shadow.shape[0], 1))))

    #concatinate them
    X_Train_Attack = np.vstack((X_Train_Shadow, X_Out_Shadow))

    #shuffle them
    np.random.shuffle(X_Train_Attack)

    #remove and take labels
    labels = X_Train_Shadow[:, -1]  # take last column
    X_Train_Attack = X_Train_Attack[:, :-1]  # take everything but the last column
    return X_Train_Attack, labels

def preform_attack_centralised(config:Config, D_Shadow, target_model, shadow_model, attack_model, X, y, fName=None) -> np.array():
    """_summary_

    Args:
        config (Config, D_Shadow, target_model, shadow_model, attack_model, X, y, fName, optional): _description_. Defaults to None)->np.array(.
    """
    D_Train_Shadow, D_Out_Shadow, D_Test = split_shadow(D_Shadow)
    
    y_pred=None
    if type(shadow_model) != SFXGBoost:
        y_pred = shadow_model.fit(D_Train_Shadow, D_Train_Shadow).predict(D_Test)
    else:
        y_pred = shadow_model.fit(D_Train_Shadow, D_Train_Shadow, fName).predict(D_Test)
    
    Metric_shadow_acc = accuracy_score(D_Test[1], y_pred)
    del y_pred
    
    z, label = f_random(D_Train_Shadow, D_Out_Shadow)
    z_train, z_test, label_train, label_test = train_test_split(z, label, train_size=0.8, test_size=0.2, random_state=12)


    attack_model.fit(z_train, label_train)
    
    y_pred = attack_model.predict(z_test)
    Metric_attack_acc = accuracy_score(label_test, y_pred)
    Metric_attack_precision = precision_score(label_test, y_pred)
    print(f"DEBUG: accuracy attack: {Metric_attack_acc}")



