from logging import Logger
from SFXGBoost.config import Config, MyLogger
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from SFXGBoost.Model import SFXGBoost
from SFXGBoost.data_structure.treestructure import FLTreeNode
from sklearn.model_selection import train_test_split
from SFXGBoost.common import flatten_list
from SFXGBoost.common.pickler import retrieve
import sys

def split_shadow(D_Shadow):
    """Splits the shadow_dataset such that it can be used for training with D_Train_Shadow and D_Out_Shadow. 

    Args:
        D_Shadow Tuple(nd.array): a Tuple with two arrays for X and y. y is One Hot encoded. 

    Returns:
        Tuple(nd.array), Tuple(ndarray): D_Train_Shadow, D_Out_Shadow
    """
    X = D_Shadow[0]
    y = D_Shadow[1]
    # print(np.shape(X))
    split = len(X[:, 0]) // 2 # find what one third of the users are
    D_Train_Shadow = (X[:split, :], y[:split, :])
    D_Out_Shadow = (X[split:, :], y[split:, :])

    return D_Train_Shadow, D_Out_Shadow

def federated_split(D_Shadow):
    num_shadows = len(D_Shadow)
    D_Shadow[0:num_shadows-1], D_Shadow[num_shadows]

def f_random(D_Train_Shadow, D_Out_Shadow, seed=0, take=0):
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
    if take != 0:
        take = take//2 # TODO investigate centralised!
        if seed != 0:
            np.random.seed(int(seed))
        take = min(take, min_lenght) # don't take more than you can
        indices = np.random.choice(min_lenght, int(take), replace=False)
        X_Train_Shadow = X_Train_Shadow[indices]
        X_Out_Shadow = X_Out_Shadow[indices]

    # add an extra column with 1 if Train_Shadow else 0
    X_Train_Shadow = np.hstack( (X_Train_Shadow, np.ones((X_Train_Shadow.shape[0], 1)))) 
    X_Out_Shadow = np.hstack( (X_Out_Shadow, np.zeros((X_Out_Shadow.shape[0], 1))))

    #concatinate them
    X_Train_Attack = np.vstack((X_Train_Shadow, X_Out_Shadow))

    #shuffle them
    # np.random.seed(seed)
    np.random.shuffle(X_Train_Attack)

    #remove and take labels
    labels = X_Train_Attack[:, -1]  # take last column
    X_Train_Attack = X_Train_Attack[:, :-1]  # take everything but the last column
    # print(f"labels = {labels.shape}")
    # print(f"X_Train_Attack = {X_Train_Attack.shape}")
    return X_Train_Attack, labels

def create_D_attack_centralised(shadow_model_s, D_Train_Shadow, D_Out_Shadow, config:Config=None):
    # Shadow_model_s can be multiple shadow_models! TODO deal with that!
    x, labels = f_random(D_Train_Shadow, D_Out_Shadow)
    z = None
    if type(shadow_model_s) == SFXGBoost:
        print(f"target rank = {config.target_rank}")
        z = shadow_model_s.predict_proba(x, config.target_rank)
    else:
        z = shadow_model_s.predict_proba(x)
    # z_top_indices = np.argsort(z)[::-1][:3] # take top 3 sorted by probability
    # z = np.take(z, z_top_indices) # take top 3
    return z, labels

def get_G_H(config:Config, node):
    if config.target_rank != 0:
        return flatten_list([node.Gpi[config.target_rank-1], node.Hpi[config.target_rank-1]])
    else:
        return flatten_list([node.G, node.H])

def get_info_node(config:Config, x:np.array, node=None):
    # for node in nodes:
    splittinginfo = []
    parent:FLTreeNode = node.parent
    curnode = node
    i = 0
    if parent is None: # root node!
        input = np.column_stack((x, np.tile(get_G_H(config, node), ((x.shape[0]), 1)) ))
        return input
    while parent != None:
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

    input = np.column_stack((x, np.tile(splittinginfo, (x.shape[0], 0)), np.tile(get_G_H(config, node), ((x.shape[0]), 1)) ))  # copy the flattened information for every x, f(x)=z
    return input

def get_train_Attackmodel_1(config:Config, logger:Logger, n_shadows, c, d, D_Train_Shadow): # TODO make faster :(
    input = None
    D_Train_Attack = None  # c, l empty lists size is nshadows * (len(D_train_shadow)*2) * n_nodes # where n_nodes scales exponentially
    y=[]
    max_length = 50_000 # try to devide the max_lenght over the different nodes over the available trees. 
    max_length_shadow = max_length//n_shadows

    #TODO take randomly from t, node, then add all NO DUPES x's
    #     
    for a in range(n_shadows):
        logger.warning(f"attack1 creation shadow model {a} out of {n_shadows} for c={c}, d={d}")
        shadow_model:SFXGBoost = retrieve("shadow_model_" + str(a), config)
        nodes = np.array(shadow_model.nodes[c][d])  # can be zero
        if len(nodes) == 0:
            max_length_shadow += max_length_shadow//(n_shadows-a) # devide the nodes to be put here over the other nodes of other shadow models
            continue
        x_len = len(D_Train_Shadow[a][0]) + len(D_Train_Shadow[(a+2)%len(D_Train_Shadow)][0])
        
        wish_to_take = int(np.min((max_length_shadow, x_len, len(nodes))))
        
        # TODO take 1, but make it unique such that x is not used twice, not vital but could skew results a tiny bit. 

        x, labels_train = f_random(D_Train_Shadow[a], D_Train_Shadow[(a+2) % len(D_Train_Shadow)], seed=a+c+d, take=wish_to_take) #int(max_length_shadow_tree_node)
        
        indices = np.random.choice(int(len(nodes)), len(x), replace=False)
        random_nodes = nodes[indices]

        for nodepos, node in enumerate(random_nodes):
            # z = shadow_model.predict_proba(x)
            input = get_info_node(config, np.array([x[nodepos]]), node)  # take one of the x's linked to the node, and get the G,H
            y.append(labels_train[nodepos])  # append would be nicer, 

            if not D_Train_Attack is None: 
                D_Train_Attack = np.vstack((D_Train_Attack, input))
            else:
                D_Train_Attack = input
    return D_Train_Attack, np.array(y)  # .reshape(-1, 1)

def get_input_attack2(config:Config, D_Train_Shadow, n_models, attack_models):            
    input_attack_2 = []
    y_attack_2 = []
    model, D_in, D_out = None, None, None
    for a, model in range(n_models):
        if n_models == 1:
            model = retrieve("target_model", config)
            D_in = D_Train_Shadow[0]
            D_out = D_Train_Shadow[1]
        else:
            model = retrieve("shadow_model_" + str(a), config)
            D_in = D_Train_Shadow[(a+1) % len(D_Train_Shadow)]
            D_out = D_Train_Shadow[(a+3) % len(D_Train_Shadow)]

        print(f"busy with model {a} out of {n_models}")

        x, labels_train = f_random(D_in, D_out, seed=a)
        # z = model.predict_proba(x)#[:, 1]
        # input = np.column_stack( (x, z) )
        input = None  # no x and z

        for c in range(config.nClasses):
            for d in range(config.max_depth):
                all_p = None
                # nodes = model.trees[c][t].root.get_nodes_depth(d)
                nodes = model.nodes[c][d]
                if nodes == []:
                    continue
                for node in nodes:
                    
                    node_params = get_info_node(config, x ,node) #size = 2000
                    list_of_outputs = attack_models[c][d].predict_proba(node_params)[:, 1] # returns two dimentional array!!!!
                    
                    if all_p is None:
                        all_p = list_of_outputs
                    else:
                        all_p = np.column_stack((all_p, list_of_outputs)) # all probas for every x
                
                if all_p is None:
                    print(f"Warning, at depth {d} of model {a} no nodes were found accross all trees of class {c}?!")
                    if (input is None):
                        input = np.zeros( (x.shape[0], 1) )
                    else:
                        input = np.column_stack((input, np.zeros( (x.shape[0],1) )))
                else:
                    avg = np.average(all_p, axis=1).reshape((-1,1))
                    # min = np.min(all_p, axis=1).reshape((-1,1))
                    # max = np.max(all_p, axis=1).reshape((-1,1))
                    if (input is None):
                        input = avg
                        # input = np.column_stack((avg))
                    else:
                        input = np.column_stack((input, avg))
                    # input = np.column_stack((avg, min , max))
        if input_attack_2 == []:
            input_attack_2 = input
            y_attack_2 = labels_train
        else:
            input_attack_2 = np.vstack((input_attack_2, input))
            y_attack_2 = np.hstack((y_attack_2, labels_train))

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
    # print(f"DEBUG: {y_pred}")
    # logger.warning("DEBUG: y_pred = {y_pred}")
    # logger.warning(f"DEBUG: f(x) target = {target_model.predict_proba(test_x)}")



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
            tmp = self.model(x)
            tmp = tmp[:, 0]
            return tmp
        
        
        def fit(self, X, y, num_epochs=3, lr=0.03, batch_size=1):
            # criterion = nn.CrossEntropyLoss()
            criterion = nn.BCELoss()  # Binary Cross Entropy 
            
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            # for x in X:
                # print(x)
                # print(np.shape(np.array(x)))

            print(np.shape(X))
            print(np.shape(y))
            tensorx = torch.from_numpy(X).to(torch.float32)
            tensory = torch.from_numpy(y).to(torch.float32)
            train_data = DataLoader(TensorDataset(tensorx, tensory), batch_size, shuffle=True)
            self.train()
            epoch_loss = 0.0
            for epoch in range(num_epochs): # total pass through data
                for i, data in enumerate(train_data):
                    X_batch = Variable(data[0])
                    y_batch = Variable(data[1]).reshape((1,))
                    optimizer.zero_grad()
                    outputs = self.forward(X_batch)
                    # print(outputs)
                    # print(outputs.shape)
                    # print(y_batch)
                    # print(y_batch.shape)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss
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