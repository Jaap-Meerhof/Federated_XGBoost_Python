from SFXGBoost.config import Config, rank, comm, MyLogger
from SFXGBoost.Model import PARTY_ID, SFXGBoostClassifierBase, SFXGBoost
from SFXGBoost.data_structure.databasestructure import QuantiledDataBase, DataBase
from SFXGBoost.MemberShip import preform_attack_centralised, split_shadow, create_D_attack_centralised
from SFXGBoost.common.pickler import *
import SFXGBoost.view.metric_save as saver
from SFXGBoost.dataset.datasetRetrieval import getDataBase

import numpy as np
from logging import Logger
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import torch
import sys

def log_distribution(logger, X_train, y_train, y_test):
    nTrain = len(y_train)
    nZeroTrain = np.count_nonzero(y_train == 0)
    rTrain = nZeroTrain/nTrain

    nTest = len(y_test)
    nZeroTest = np.count_nonzero(y_test == 0)
    rTest = nZeroTest / nTest
    logger.warning("DataDistribution, nTrain: %d, zeroRate: %f, nTest: %d, ratioTest: %f, nFeature: %d", 
    nTrain, rTrain, nTest, rTest, X_train.shape[1])

def fit(X_train, y_train, X_test, fName, model):
    quantile = QuantiledDataBase(DataBase.data_matrix_to_database(X_train, fName) )

    initprobability = (sum(y_train))/len(y_train)
    total_users = comm.Get_size() - 1
    total_lenght = len(X_train)
    elements_per_node = total_lenght//total_users
    start_end = [(i * elements_per_node, (i+1)* elements_per_node) for i in range(total_users)]

    if rank != PARTY_ID.SERVER:
        start = start_end[rank-1][0]
        end = start_end[rank-1][1]
        X_train_my, X_test_my = X_train[start:end, :], X_test[start:end, :]
        y_train_my = y_train[start:end]

    # split up the database between the users
    if rank == PARTY_ID.SERVER:
        pass
        model.setData(fName=fName)
    else:
        original = DataBase.data_matrix_to_database(X_train_my, fName)
        quantile = quantile.splitupHorizontal(start_end[rank-1][0], start_end[rank-1][1])
        model.setData(quantile, fName, original, y_train_my)
    
    model.boost(initprobability)

    
    if rank == PARTY_ID.SERVER:
        y_pred = model.predict(X_test)
    else:
        y_pred = [] # basically a none

    y_pred_org = y_pred.copy()
    X = X_train
    y = y_train 
    return X, y, y_pred_org

def test_global(config:Config, logger:Logger, model: SFXGBoostClassifierBase, getDatabaseFunc):
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDatabaseFunc()
    log_distribution(logger, X_train, y_train, y_test)
    xgboostmodel = None
    if rank == PARTY_ID.SERVER:
        import xgboost as xgb
        xgboostmodel = xgb.XGBClassifier(ax_depth=config.max_depth, objective="multi:softmax", tree_method="approx",
                            learning_rate=0.3, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=0, reg_lambda=config.lam)
        xgboostmodel.fit(X_train, np.argmax(y_train, axis=1))
        from sklearn.metrics import accuracy_score
        print(np.shape(X_test))

        y_pred_xgb = xgboostmodel.predict(X_test)
        print(f"Accuracy xgboost normal = {accuracy_score(np.argmax(y_test, axis=1), y_pred_xgb)}")
    
    X, y, y_pred_org = fit(X_train, y_train, X_test, fName, model)
    # X, y = X_train, y_train
    # y_pred_org = xgboostmodel.predict(X_test)
    # model = xgboostmodel
    # TODO make it such that model.fit gets used instead. its more clear and easy! 
    
    return X, y, y_pred_org, y_test, model, X_shadow, y_shadow, fName

dataset_list = ['purchase-10', 'purchase-20', 'purchase-50', 'purchase-100', 'texas', 'healthcare', 'MNIST', 'synthetic', 'Census', 'DNA']
POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]

def get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger):
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS)()
    log_distribution(logger, X_train, y_train, y_test)

    target_model.fit(X_train, y_train, fName)
    shadow_model.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName)
    D_Train_Shadow, D_Out_Shadow, D_Test = split_shadow((X_shadow, y_shadow))
    z, labels = create_D_attack_centralised(shadow_model, D_Train_Shadow, D_Out_Shadow)
    attack_model.fit(z, labels)
    z_test, labels_test = create_D_attack_centralised(target_model, (X_train, y_train), D_Test)
    
    data = retrieve_data(target_model, shadow_model, attack_model, X_train, y_train, X_test, y_test, z_test, labels_test)
    return data

def train_all_federated(target_model, shadow_models, attack_model, config:Config, logger:MyLogger) -> dict:
    """Trains for a federated attack given the target_model, shadow_model(s), attack_model, config, logger

    Args:
        target_model (any): must support .fit() & .predict().or must be SFXGBoost.
        shadow_models (any): must support .fit() & .predict().or must be SFXGBoost.
        attack_model (any): must support .fit() & .predict().or must be SFXGBoost
        config (Config): config with experiment variables
        logger (MyLogger): logs the whole thing

    Returns:
        dict: dict that stores the different metrics, check retrieve_data for which metrics
    """
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS)()
    log_distribution(logger, X_train, y_train, y_test)

    target_model.fit(X_train, y_train, fName)

    for shadow_model in shadow_models:

        D_Train_Shadow, D_Out_Shadow, D_Test = split_shadow((X_shadow, y_shadow))
        shadow_model.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName)
        shadow_model.retrieve_differentials()  # TODO implement, this will retrieve the gradients & hessians for the attack model
    
    attack_model = create_attack_model_federated()

    attack_model.fit(z, labels)

    # create z_test and labels_test 
    # z test is the | X_train + differentials --> label 1
    #               | X_test + differentials --> label 0

    data = retrieve_data(target_model, shadow_model, attack_model, X_train, y_train, X_test, y_test, z_test, labels_test)
    return data

def retrieve_data(target_model, shadow_model, attack_model, X_train, y_train, X_test, y_test, z_test, labels_test): # TODO put this in SFXGBoost
    data = {}
    from sklearn.metrics import accuracy_score, precision_score

    y_pred_train_target = target_model.predict(X_train)                 
    y_pred_test_target = target_model.predict(X_test)                   
    acc_train_target = accuracy_score(y_train, y_pred_train_target)     # accuarcy train target
    data["accuracy train target"] = acc_train_target                         # accuracy train target
    prec_train_target = precision_score(y_train, y_pred_train_target)
    data["precision train target"] = prec_train_target
    acc_test_target = accuracy_score(y_test, y_pred_test_target)
    data["accuray test target"] = acc_test_attack
    prec_test_target = precision_score(y_test, y_pred_test_target)
    data["precision test target"] = prec_test_target
    overfit_target = acc_train_target - acc_test_target
    data["overfitting target"] = overfit_target

    y_pred_train_shadow = shadow_model.predict(X_train)
    y_pred_test_shadow = shadow_model.predict(X_test)
    acc_train_shadow = accuracy_score(y_train, y_pred_train_shadow)
    data["accuracy train shadow"] = acc_train_shadow
    prec_train_shadow = precision_score(y_train, y_pred_train_shadow)
    data["precision train shadow"] = prec_train_shadow
    acc_test_shadow = accuracy_score(y_test, y_pred_test_shadow)
    data["accuracy test shadow"] = acc_test_shadow
    prec_test_shadow = precision_score(y_test, y_pred_test_shadow)
    data["precision test shadow"] = prec_test_shadow
    overfit_shadow = acc_train_shadow - acc_test_shadow
    data["overfitting shadow"] = overfit_shadow

    y_pred_test_attack = attack_model.predict(z_test) # true = labels_test
    acc_test_attack = accuracy_score(labels_test, y_pred_test_attack)
    data["accuracy test attack"] = acc_test_attack
    prec_test_attack = precision_score(labels_test, y_pred_test_attack)
    data["precision test attack"] = prec_test_attack
    return data

def create_attack_model_federated(config:Config, G, H):

    nFeatures = len(G[0])
    nTrees = len(G)
    max_depth = config.max_depth
    max_tree = config.max_tree

    import torch.nn as nn
    class CNN(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(CNN, **kwargs).__init__()
            
            self.linear1 = nn.Linear()
            self.linear2 = nn.Linear()
            self.softmax = nn.Softmax()

            self.differential_cov = [nn.Conv2d(stride=1) for depth in range(config.max_depth)] # TODO determine what the convolution should look like
            self.dropout = nn.Dropout(p=0.2)
            self.differential_lin2 = [nn.Linear() for depth in range(config.max_depth)]
            self.f = nn.Sequential(
                nn.Linear(100, 100), # TODO 
                nn.ReLU(),
                nn.Dropout(p=0.2)
            )

        def forward(self, x): 
            # TODO add correcty data input
            return self.f(x)
        
        def fit(self, x, y, num_epochs, lr):
            criterion = nn.MSELoss
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

            for epoch in range(num_epochs):
                pass

    return CNN # returns the class


def experiment2():
    seed = 10

    datasets = ["healthcare", "MNIST", "synthetic", "Census", "DNA", "Purchase-10", "Purhase-20", "Purchase-50", "Purchase-100"]
    targetArchitectures = ["XGBoost", "FederBoost-central", "FederBoost-federated"]
    all_data = {} # all_data["XGBoost"]["healthcore"]["accuarcy train shadow"]

    for targetArchitecture in targetArchitectures:
        all_data[targetArchitecture] = {}
        for dataset in datasets:
            config = Config(experimentName = "experiment 2",
            nameTest= dataset + " test",
            model="normal",
            dataset=dataset,
            lam=0.1, # 0.1 10
            gamma=0.5,
            alpha=0.0,
            learning_rate=0.3,
            max_depth=5,
            max_tree=9,
            nBuckets=100)
            logger = MyLogger(config).logger
            logger.warning(config.prettyprint())
            np.random.RandomState(seed) # TODO set seed sklearn.split
            
            if targetArchitecture == "Federboost-central":
                target_model = SFXGBoost(config, logger)
                shadow_model = SFXGBoost(config, logger)
                attack_model = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
                data = get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger)
                all_data[targetArchitecture][dataset] = data
            elif targetArchitecture == "FederBoost-federated":
                target_model = SFXGBoost(config, logger)
                shadow_models = [SFXGBoost(config, logger) for _ in range(10)]  # 
                attack_model= None  # TODO define federated neural network.
                data = train_all_federated(target_model, shadow_models, attack_model, config, logger)
                all_data[targetArchitecture][dataset] = data
            elif targetArchitecture == "XGBoost":  # define neural network
                target_model = xgb.XGBClassifier(ax_depth=config.max_depth, objective="multi:softmax", tree_method="approx",
                        learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
                shadow_model = xgb.XGBClassifier(ax_depth=config.max_depth, objective="multi:softmax", tree_method="approx",
                        learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
                attack_model = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
                data = get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger)
                all_data[targetArchitecture][dataset] = data
            else:
                raise Exception("Wrong model types given!")
            
            # saver.save_experiment_2_one_run(attack_model, )

    saver.create_plots_experiment_2(all_data)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <exeperiment number>")
    # experimentNumber = int(sys.argv[1])
    dataset = "texas"
    config = Config(experimentName="tmp",
           nameTest= dataset + "0.3 20 trees",
           model="normal",
           dataset=dataset,
           lam=0.1, # 0.1 10
           gamma=0.5,
           alpha=0.0,
           learning_rate=0.3,
           max_depth=8,
           max_tree=20,
           nBuckets=100)
    logger = MyLogger(config).logger
    if rank ==0 : logger.debug(config.prettyprint())
    if config.model == "normal":
        target_model = SFXGBoost(config, logger)
        shadow_model = SFXGBoost(config, logger)
        # shadow_model = xgb.XGBClassifier(ax_depth=config.max_depth, objective="multi:softmax", tree_method="approx",
        #                 learning_rate=0.3, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=0, reg_lambda=config.lam)
        # attack_model = xgb.XGBClassifier(tree_method="exact", objective='binary:logistic', max_depth=10, n_estimators=30, learning_rate=0.3)
        # attack_model = DecisionTreeClassifier(max_depth=6,max_leaf_nodes=10)
        attack_model = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
    
    # if isSaved(config.nameTest, config):
    #     shadow_model = retrieve("model", config)
    # TODO target_model = train_model()
    # TODO shadow_model = train_model()
    # that way I can save the model reuse it and apply different attack_models on it.
    # TODO SFXGBoost().getGradients.
    
    # X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS)()
    
    # log_distribution(logger, X_train, y_train, y_test)
    # target_model.fit(X_train, y_train, fName)
    # D_Train_Shadow, D_Out_Shadow, D_Test = split_shadow((X_shadow, y_shadow))
    # shadow_model.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName)

    # z, labels = create_D_attack_centralised(shadow_model, D_Train_Shadow, D_Out_Shadow)

    # attack_model.fit(z, labels)

    # save_metrics_one_run(target_model, shadow_model, attack_model, (X_train, y_train), (X_test, y_test), D_Train_Shadow, (z, labels), D_Test)

    X, y, y_pred_org, y_test, target_model, X_shadow, y_shadow, fName = test_global(config, logger, target_model, getDataBase(config.dataset, POSSIBLE_PATHS))
    
    preform_attack_centralised(config, logger, (X_shadow, y_shadow), target_model, shadow_model, attack_model, X, y, fName)

    if rank == 0:
        from sklearn.metrics import accuracy_score
        print(f"y_test = {np.argmax(y_test, axis=1)}")
        print(f"y_pred_org = {y_pred_org}")
        print(f"Accuracy federboost = {accuracy_score(np.argmax(y_test, axis=1), y_pred_org)}")
main()