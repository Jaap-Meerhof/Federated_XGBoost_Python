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

def preform_attack_centralised(config:Config, D_Shadow, target_model, shadow_model, attack_model, X, y, fName=None) -> np.ndarray:
    """_summary_

    Args:
        config (Config, D_Shadow, target_model, shadow_model, attack_model, X, y, fName, optional): _description_. Defaults to None)->np.array(.
    """
    D_Train_Shadow, D_Out_Shadow, D_Test = split_shadow(D_Shadow)
    
    y_pred=None
    if type(shadow_model) != SFXGBoost:
        y_pred = shadow_model.fit(D_Train_Shadow[0], np.argmax(D_Train_Shadow[1], axis=1)).predict(D_Test[0])
    else:
        y_pred = shadow_model.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName).predict(D_Test[0])
    
    Metric_shadow_acc = accuracy_score(np.argmax(D_Test[1], axis=1), y_pred)
    del y_pred
    
    x, label = f_random(D_Train_Shadow, D_Out_Shadow)
    
    X_train, X_test, label_train, label_test = train_test_split(x, label, test_size=0.2, random_state=12)
    z_train = shadow_model.predict_proba(X_train)
    z_test = target_model.predict_proba(X) # todo test data outside
    test_x, test_label = f_random((X,y), D_Test)
    attack_model.fit(z_train, label_train)
    
    y_pred = attack_model.predict(target_model.predict_proba(test_x))
    Metric_attack_acc = accuracy_score(test_label, y_pred)
    print(f"DEBUG: accuracy attack: {Metric_attack_acc}")
    Metric_attack_precision = precision_score(test_label, y_pred)
    print(f"DEBUG: precision attack: {Metric_attack_precision}")
    print(f"DEBUG: {y_pred}")



