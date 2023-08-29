from logging import Logger
from SFXGBoost.config import Config, MyLogger
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from SFXGBoost.Model import SFXGBoost
from SFXGBoost.data_structure.treestructure import FLTreeNode
from sklearn.model_selection import train_test_split
from SFXGBoost.common import flatten_list
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

def fitAttackmodel(config, attack_model, shadow_models, c, l, D_Train_Shadow):
    for i, shadow_model in enumerate(shadow_models):
        D_Train_Attack = None  # c, l empty lists
        # print(f"working on D_Train for shadow model {i} out of {len(D_Train_Shadow)}")
        # shadow_model = shadow_model[i]
        x, labels_train = f_random(D_Train_Shadow[i], D_Train_Shadow[(i+1) % len(D_Train_Shadow)])  #
        y = []
        z = shadow_model.predict_proba(x)
        for t in range(config.max_tree):
            nodes = shadow_model.trees[c][t].root.get_nodes_depth(l)
            for node in nodes:
                splittinginfo = []
                parent:FLTreeNode = node.parent
                curnode = node
                i = 0
                while parent != None:
                    i += 1
                    if i > 1000:
                        raise(Exception("loop in parent?"))
                    id = parent.splittingInfo.featureId
                    splitvalue = parent.splittingInfo.splitValue
                    splittinginfo.append(id)
                    splittinginfo.append(splitvalue)
                    if parent.leftBranch == curnode:
                        pass  # smaller equal <=
                        splittinginfo.append(0)
                    elif parent.rightBranch == curnode:
                        pass  # Greater > 
                        splittinginfo.append(1)
                    else:
                        raise(Exception("??"))  # This actually caught a bug
                    curnode = parent
                    parent = parent.parent

                flattened = flatten_list([splittinginfo , node.G, node.H])  # flatten the information
                input = np.column_stack((x, z, np.tile(flattened, (x.shape[0] , 1))))  # copy the flattened information for every x, f(x)=z
                y.extend(labels_train)
                if not D_Train_Attack is None: 
                    D_Train_Attack = np.vstack((D_Train_Attack, input))
                else:
                    D_Train_Attack = input
    print(np.shape(D_Train_Attack))
    return attack_model.fit(D_Train_Attack, np.array(y).reshape(-1, 1))

def create_D_attack_federated(config: Config, D_Train_Shadow, X_train, X_test, shadow_models, target_model):
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
    from concurrent.futures import ThreadPoolExecutor
    from concurrent import futures
    # D_train_attack should be D_train_shadow label = 1, D_Out_Shadow = 0
    # I first need to combine the shadowmodel's data, then add the relevant split information to that attack model
    D_Train_Attack = [ [[] for _ in range(config.max_depth)] for _ in range(config.nClasses)]  # c, l empty lists
    D_Train_Attack2 = [ [[] for _ in range(config.max_depth)] for _ in range(config.nClasses)]  # c, l empty lists
    D_Test_Attack = [ [[] for _ in range(config.max_depth)] for _ in range(config.nClasses)]  # c, l empty lists
    labels_train, labels_train_2 = None, None
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_D_Train_Attack = [executor.submit(shadow_model_extraction, config, D_Train_Shadow, shadow_models[i], i) for i in range(len(D_Train_Shadow))  ]
        for future in futures.as_completed(future_D_Train_Attack):
            D_Train_Attack_current, D_Train_Attack2_current, labels_train_current, labels_train_2_current = future.result()
            [ [[D_Train_Attack[c][l].append(D_Train_Attack_current[c][l])] for l in range(config.max_depth)] for c in range(config.nClasses)]
            [ [[D_Train_Attack2[c][l].append(D_Train_Attack2_current[c][l])] for l in range(config.max_depth)] for c in range(config.nClasses)]

            # np.concatenate((D_Train_Attack, D_Train_Attack_current), axis=2)
            # np.concatenate((D_Train_Attack2, D_Train_Attack2_current), axis=2)
            np.hstack((labels_train, labels_train_current))
            np.hstack((labels_train_2, labels_train_2_current))
    lastlen = -1
    x, labels_test = f_random((X_train, None), (X_test, None) )
    z = target_model.predict_proba(x)
    for c in range(config.nClasses):
        print(f"working on D_Test for class {c} out of {config.nClasses}")
        for t in range(config.max_tree):
            for l in range(config.max_depth):
                nodes = target_model.trees[c][t].root.get_nodes_depth(l)
                for node in nodes:
                    splittinginfo = []
                    curnode = node
                    parent:FLTreeNode = node.parent
                    while parent != None:
                        splittinginfo.append(parent.splittingInfo.featureId)
                        splittinginfo.append(parent.splittingInfo.splitValue)
                        if parent.leftBranch == curnode:
                            splittinginfo.append(0)
                        elif parent.rightBranch == curnode:
                            splittinginfo.append(1)
                        else:
                            raise(Exception("??"))
                        curnode = parent
                        parent = parent.parent

                    flattened = flatten_list([splittinginfo , node.G, node.H])  # flatten the information
                    input = np.column_stack((x, z, np.tile(flattened, (x.shape[0] , 1))))  # copy the flattened information for every x, f(x)=z
                    D_Test_Attack[c][l].append(input)
                    
    D_Train_Attack = (D_Train_Attack, labels_train)
    D_Train_Attack2 = (D_Train_Attack2, labels_train_2)
    D_Test_Attack = (D_Test_Attack, labels_test)

    return D_Train_Attack, D_Train_Attack2, D_Test_Attack

def shadow_model_extraction(config, D_Train_Shadow, shadow_model, i):
    D_Train_Attack = [ [[] for _ in range(config.max_depth)] for _ in range(config.nClasses)]  # c, l empty lists
    D_Train_Attack2 = [ [[] for _ in range(config.max_depth)] for _ in range(config.nClasses)]  # c, l empty lists
    print(f"working on D_Train for shadow model {i} out of {len(D_Train_Shadow)}")
    # shadow_model = shadow_model[i]
    x, labels_train = f_random(D_Train_Shadow[i], D_Train_Shadow[(i+1) % len(D_Train_Shadow)])  #
    z = shadow_model.predict_proba(x)
    x_2, labels_train_2 = f_random(D_Train_Shadow[(i+2) % len(D_Train_Shadow)], D_Train_Shadow[(i+3) % len(D_Train_Shadow)])
    z_2 = shadow_model.predict_proba(x_2)
    for c in range(config.nClasses):
        # print(f"class {c} out of {config.nClasses}")
        for t in range(config.max_tree):
            # print(f"tree {t} out or {config.max_tree}")
            for l in range(config.max_depth):
                # print(f"depth {l} out of {config.max_depth}")
                nodes = shadow_model.trees[c][t].root.get_nodes_depth(l)
                for node in nodes:
                    splittinginfo = []
                    parent:FLTreeNode = node.parent
                    curnode = node
                    i = 0
                    while parent != None:
                        i += 1
                        if i > 1000:
                            raise(Exception("loop in parent?"))
                        id = parent.splittingInfo.featureId
                        splitvalue = parent.splittingInfo.splitValue
                        splittinginfo.append(id)
                        splittinginfo.append(splitvalue)
                        if parent.leftBranch == curnode:
                            pass  # smaller equal <=
                            splittinginfo.append(0)
                        elif parent.rightBranch == curnode:
                            pass  # Greater > 
                            splittinginfo.append(1)
                        else:
                            raise(Exception("??"))  # This actually caught a bug
                        curnode = parent
                        parent = parent.parent

                    flattened = flatten_list([splittinginfo , node.G, node.H])  # flatten the information
                    input = np.column_stack((x, z, np.tile(flattened, (x.shape[0] , 1))))  # copy the flattened information for every x, f(x)=z
                    D_Train_Attack[c][l].append(input)
                    input_2 = np.column_stack((x_2, z_2, np.tile(flattened, (x.shape[0] , 1))))  # copy the flattened information for every x, f(x)=z
                    D_Train_Attack2[c][l].append(input_2)

    return D_Train_Attack, D_Train_Attack2, labels_train, labels_train_2

def get_input_attack2(config:Config, D_train_Attack, attack_models):
    input_attack_2 = []
    y_attack_2 = []
    for c in range(config.nClasses):
        x_one = []
        for l, attack_model in enumerate(attack_models[c]):
            labels = D_train_Attack[1][c][l]
            preds = attack_model.predict(D_train_Attack[0][c][l])
            max = np.max(preds, axis=1, initial=0)
            min = np.min(preds, axis=1, initial=0)
            avg = np.average(preds, axis=1, initial=0)
            list.extend([max, min, avg])
        input_attack_2.append(x_one)
        y_attack_2.append(labels)
    return (input_attack_2, y_attack_2)

def predict_federated(config:Config, attack_models, Attack_Model_2, D_test_Attack):
    """predicts on the attack_models, using D_test_Attack, this is fed into Attack_Model_2.

    Args:
        attack_models (list[list[SFXGBoost]]): The attack model's for every class, for every layer 
        Attack_Model_2 (any): the overarching attack_model that takes the output of the attack_models
        D_test_Attack (tuple[list, list]): test datset for attack_models
    """
    for c in config.nClasses:
        for l in config.max_depth:
            attack_models[c][l].predict(D_test_Attack[c][l])


    return

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



import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader, TensorDataset

class DepthNN(nn.Module):
        def __init__(self, input_Features, *args, **kwargs) -> None:
            super(DepthNN, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(input_Features, input_Features), # TODO 
                nn.ReLU(),
                nn.Dropout(p=0.2), 
                nn.Linear(input_Features, 2),
                nn.Softmax(dim=1)
            )

        def forward(self, x): 
            # TODO add correcty data input
            return self.model(x)
        
        def fit(self, X, y, num_epochs=1000, lr=0.03, batch_size=1):
            # criterion = nn.CrossEntropyLoss()
            criterion = nn.BCELoss()  # Binary Cross Entropy 
            
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            # for x in X:
                # print(x)
                # print(np.shape(np.array(x)))

            print(np.shape(X))
            print(np.shape(y))
            train_data = DataLoader(TensorDataset(np.array(X), np.array(y)), batch_size, shuffle=True)
            self.train()
            epoch_loss = 0.0
            for epoch in range(num_epochs): # total pass through data
                for i, data in enumerate(train_data):
                    X_batch = Variable(data[0])
                    y_batch = Variable(data[1])
                    optimizer.zero_grad()
                    outputs = self.forward(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.data[0]
                    # float(epoch_loss) / (i+1)
            
        def predict(self, X, batchsize=1): # https://github.com/henryre/pytorch-fitmodule/blob/master/pytorch_fitmodule/fit_module.py
            self.eval()
            data = DataLoader(TensorDataset(X, torch.Tensor(X.size()[0])), batch_size=1, shuffle=False)
            r, n = 0, X.size()[0]
            for batch_data in data:
                X_batch = Variable(batch_data)
                y_batch_pred = self(X_batch).data
                if r == 0:
                    y_pred = torch.zeros((n, ) + y_batch_pred.size()[1:])
                y_pred[r : min(n, r + batchsize)] = y_batch_pred
                r+= batchsize
            return y_pred