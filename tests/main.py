from SFXGBoost.config import Config, rank, comm, MyLogger
from SFXGBoost.Model import PARTY_ID, SFXGBoostClassifierBase, SFXGBoost, devide_D_Train
from SFXGBoost.data_structure.databasestructure import QuantiledDataBase, DataBase
from SFXGBoost.MemberShip import preform_attack_centralised, split_shadow, create_D_attack_centralised, predict_federated, get_input_attack2, DepthNN, get_train_Attackmodel_1 
from SFXGBoost.common.pickler import *
import SFXGBoost.view.metric_save as saver
from SFXGBoost.dataset.datasetRetrieval import getDataBase
from SFXGBoost.view.plotter import plot_experiment2, plot_auc
from SFXGBoost.view.table import create_latex_table_1, create_table_config
from typing import List
import numpy as np
from logging import Logger
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
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
        xgboostmodel = xgb.XGBClassifier(max_depth=config.max_depth, objective="multi:softmax", tree_method="approx",
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
    
    return X, y, X_test, y_test, y_pred_org, model, X_shadow, y_shadow, fName

dataset_list = ['purchase-10', 'purchase-20', 'purchase-50', 'purchase-100', 'texas', 'healthcare', 'MNIST', 'synthetic', 'Census', 'DNA']
POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]

def get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger:Logger, name="sample"):
    # TODO keep track of rank 
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS)()
    log_distribution(logger, X_train, y_train, y_test)
    D_Train_Shadow, D_Out_Shadow = split_shadow((X_shadow, y_shadow)) 
    if type(target_model) == SFXGBoost:
        target_model.fit(X_train, y_train, fName) # chrashes
        shadow_model.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName)
    else:
        target_model.fit(X_train, np.argmax(y_train, axis=1))
        shadow_model.fit(D_Train_Shadow[0], np.argmax(D_Train_Shadow[1], axis=1))
    z, labels = create_D_attack_centralised(shadow_model, D_Train_Shadow, D_Out_Shadow)
    data = None
    if rank == PARTY_ID.SERVER:
        attack_model.fit(z, labels)
        z_test, labels_test = create_D_attack_centralised(target_model, (X_train, y_train), (X_test, y_test))
        
        data = retrieve_data(target_model, shadow_model, attack_model, X_train, y_train, X_test, y_test, z_test, labels_test)
        logger.warning("accuracy test attack: " + str(data["accuracy test attack"]))
        # plot_auc(labels_test, attack_model.predict_proba(z_test)[:, 1], f"Plots/experiment2_centralised_AUC_{name}.jpeg")
    return data

from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
def concurrent(config, logger, c, shadow_models, D_Train_Shadow, attack_models):
    print(f"training attack models for class {c} out of {config.nClasses}")
    for d in range(config.max_depth):
        print(f"c={c} l={d}")
        name = f"Attack_model {c},{d}"
        X_Train_Attack, y = None, None
        if not(isSaved(name, config)):
            X_Train_Attack, y = get_train_Attackmodel_1(config, logger, shadow_models, c, d, D_Train_Shadow) # TODO this is too slow, multithread?
        else:
            D_Train_Attack = retrieve(f"D_Train_Attack_{c},{d}", config)
            X_Train_Attack = D_Train_Attack[0]
            y = D_Train_Attack[1]
        attack_models[c][d] = retrieveorfit(attack_models[c][d], f"Attack_model {c},{d}", config, X_Train_Attack, np.array(y))
        save((X_Train_Attack, np.array(y)), f"D_Train_Attack_{c},{d}", config) #tmp
    return c, attack_models[c]

def concurrent_multi(c, config, logger, shadow_models, D_Train_Shadow, attack_models):
    print(f"training attack models for class {c} out of {config.nClasses}")
    for l in range(config.max_depth):
        print(f"c={c} l={l}")
        name = f"Attack_model {c},{l}"
        D_Train_Attack, y = None, None
        attack_models[c][l] = MLPClassifier(hidden_layer_sizes=(100,50,1), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
        if not(isSaved(name, config)):
            D_Train_Attack, y = get_train_Attackmodel_1(config, shadow_models, c, l, D_Train_Shadow) # TODO this is too slow, multithread?
        attack_models[c][l] = retrieveorfit(attack_models[c][l], f"Attack_model {c},{l}", config, D_Train_Attack, np.array(y))

def train_all_federated(target_model, shadow_models, attack_models1:List, attack_model2, config:Config, logger:Logger) -> dict:
    """Trains for a federated attack given the target_model, shadow_model(s), attack_model, config, logger

    Args:
        target_model (SFXGBoost): must support .fit() & .predict().or must be SFXGBoost.
        shadow_models (list[SFXGBoost]): must support .fit() & .predict().or must be SFXGBoost.
        attack_models (any): must support .fit() & .predict().or must be SFXGBoost
        config (Config): config with experiment variables
        logger (MyLogger): logs the whole thing

    Returns:
        dict: dict that stores the different metrics, check retrieve_data for which metrics
    """
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS, federated=True)()
    log_distribution(logger, X_train, y_train, y_test)

    target_model = retrieveorfit(target_model, "target_model", config, X_train, y_train, fName)
    if rank != PARTY_ID.SERVER:
        quantiles = target_model.copyquantiles()
        for shadow_model in shadow_models:
            shadow_model.copied_quantiles = quantiles  
    # D_Train_Shadow, D_Out_Shadow = split_shadow((X_shadow, y_shadow)) # TODO split shadow model to be same size as X_train
    # TODO split shadow for federated such that there is a list for all. D_Out does not have to exist in federated
    n_shadows = len(X_shadow)

    D_Train_Shadow =   [(X_shadow[i]              , y_shadow[i]              ) for i in range(n_shadows)]
    print(f"attacking target = {config.target_rank}")

    for a, shadow_model in enumerate(shadow_models):
        shadow_train_x = np.vstack((D_Train_Shadow[a][0], D_Train_Shadow[(a+1)%n_shadows][0]))
        shadow_train_y = np.vstack((D_Train_Shadow[a][1], D_Train_Shadow[(a+1)%n_shadows][1]))
        shadow_models[a] = retrieveorfit(shadow_model, "shadow_model_" + str(a), config, shadow_train_x, shadow_train_y, fName)

    # G = shadow_models[0].trees[0][0].root.G # take first tree # first feature

    data = None
    if rank == PARTY_ID.SERVER:
        # attack_models = create_attack_model_federated(config, G) # returns a attack model for every class & level attack_models[c][l]
        logger.warning("creating D_attack")
        print("creating D_attack")
        # TODO create test data for attack models! :)
        # for c in tqdm(range(config.nClasses), desc="> training attack models"):
        if config.target_rank != 0:
            D_Train_Shadow = [devide_D_Train(X_shadow[i], y_shadow[i], config.target_rank) for i in range(n_shadows) ]  # make shadow_train the same size as the user we want to attack. 
            X_train, y_train = devide_D_Train(X_train, y_train, config.target_rank)  # change D_target to be the same as actually used by the targeted user
            X_test, y_test = devide_D_Train(X_test, y_test, config.target_rank)  # change D_test to be same size as targeted user
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_D_Train_Attack = [executor.submit(concurrent, config, logger, c, deepcopy(shadow_models), deepcopy(D_Train_Shadow), deepcopy(attack_models1)) for c in range(config.nClasses)  ]

            for future in futures.as_completed(future_D_Train_Attack):
                c, attack_models1[c] = future.result()
        
        tmp_data = [D_Train_Shadow[(a+1) % len(D_Train_Shadow)] for a in range(len(D_Train_Shadow))]
        for c in range(config.nClasses):
            for d in range(config.max_depth):
                X_Train_Attack, y = get_train_Attackmodel_1(config, logger, shadow_models, c, d, tmp_data)
                save((X_Train_Attack, y), f"D_Test_Attack_{c},{d}", config)
        del tmp_data
        # after attack_models are done, train another model ontop of it.
        # attack_model_2 = MLPClassifier(hidden_layer_sizes=(100, 50,25,1), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000)

        # train attack_model 2 on the outputs of attack_models, max(p) & min(p) & avg(p) level 0,.., max(p) & min(p) & avg(p) level l. * max_tree * nClasses
        
        # (X, y)
        D_train_attack2 = None
        if isSaved("D_train_attack2", config):
            D_train_attack2 = retrieve("D_train_attack2", config)
        else:
            D_train_attack2 = get_input_attack2(config, D_Train_Shadow, shadow_models, attack_models1)
            save(D_train_attack2, "D_train_attack2", config)

        attack_model2.fit(D_train_attack2[0], D_train_attack2[1])
        print(f"fitted attack model2 !")
        # predict on attack_model -> attack_model_2 -> asses on D_test_Attack
        D_test_attack2 = (X_train, y_train), (X_test, y_test)
        # D_test_attack2 = get_input_attack2(config, D_test_attack2, [target_model], attack_models)
        if isSaved("D_test_attack2", config):
            D_test_attack2 = retrieve("D_test_attack2", config)
        else:
            D_test_attack2 = get_input_attack2(config, D_test_attack2, [target_model], attack_models1)
            save(D_test_attack2, "D_test_attack2", config)

        y_pred = attack_model2.predict(D_test_attack2[0])
        z_test = D_test_attack2[0]
        labels_test = D_test_attack2[1]

        # TODO take a couple of x's, put them into shadow model, take G & H and 

        # y_pred = predict_federated(config, attack_models, Attack_Model_2, D_test_Attack)

        # create z_test and labels_test 
        # z test is the | X_train + differentials --> label 1
        #               | X_test + differentials --> label 0

        data = retrieve_data(target_model, shadow_models, attack_model2, X_train, y_train, X_test, y_test, z_test, labels_test)
        logger.warning(f'accuracy attack = {data["accuracy test attack"]}')
        print(f'accuracy attack = {data["accuracy test attack"]}')
        logger.warning(f'precision attack= {data["precision test attack"]}')
        print(f'precision attack= {data["precision test attack"]}')
        probas = attack_model2.predict_proba(D_test_attack2[0])[:,0]
        print(np.shape(probas))
        plot_auc( labels_test , probas )
    return data

def retrieve_data(target_model, shadow_model, attack_model, X_train, y_train, X_test, y_test, z_test, labels_test): # TODO put this in SFXGBoost
    data = {}
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    y_pred_train_target = target_model.predict(X_train)                 
    y_pred_test_target = target_model.predict(X_test)                   
    acc_train_target = accuracy_score(y_train, y_pred_train_target)     # accuarcy train target
    data["accuracy train target"] = acc_train_target                         # accuracy train target
    prec_train_target = precision_score(y_train, y_pred_train_target, average="weighted")
    data["precision train target"] = prec_train_target
    acc_test_target = accuracy_score(y_test, y_pred_test_target)
    data["accuray test target"] = acc_test_target
    prec_test_target = precision_score(y_test, y_pred_test_target, average="weighted")
    data["precision test target"] = prec_test_target
    overfit_target = acc_train_target - acc_test_target
    data["overfitting target"] = overfit_target

    # not that important when we know what it is like with target. 
    # TODO can still test if I add a for loop for the shadow_models in federated testing
    # y_pred_train_shadow = shadow_model.predict(X_train)
    # y_pred_test_shadow = shadow_model.predict(X_test)
    # acc_train_shadow = accuracy_score(y_train, y_pred_train_shadow)
    # data["accuracy train shadow"] = acc_train_shadow
    # prec_train_shadow = precision_score(y_train, y_pred_train_shadow)
    # data["precision train shadow"] = prec_train_shadow
    # acc_test_shadow = accuracy_score(y_test, y_pred_test_shadow)
    # data["accuracy test shadow"] = acc_test_shadow
    # prec_test_shadow = precision_score(y_test, y_pred_test_shadow)
    # data["precision test shadow"] = prec_test_shadow
    # overfit_shadow = acc_train_shadow - acc_test_shadow
    # data["overfitting shadow"] = overfit_shadow

    y_pred_test_attack = attack_model.predict(z_test) # true = labels_test
    acc_test_attack = accuracy_score(labels_test, y_pred_test_attack)
    data["accuracy test attack"] = acc_test_attack
    prec_test_attack = precision_score(labels_test, y_pred_test_attack, average="weighted")
    data["precision test attack"] = prec_test_attack
    recall_test_attack = recall_score(labels_test, y_pred_test_attack)
    data["recall test attack"] = recall_test_attack
    return data

def create_attack_model_federated(config:Config, G:list) -> list:
    """_summary_

    Args:
        config (_type_): _description_

    Returns:
        _type_: _description_
    """    
    nFeatures = config.nFeatures
    max_depth = config.max_depth

    NBins = np.sum([len(gi) for gi in G ]) # count the amount of bins found in the first tree's Gradients which will apply to all trees and H
    Attack_Models = [ [None for _ in range(max_depth)] for _ in range(config.nClasses)]
    
    for c in range(config.nClasses):
        for l in range(max_depth):
            nInputs = nFeatures + config.nClasses + (3 * l) + NBins + NBins  # x + f(x) + (min(s0) + max(s0) + f0 * l) + G + H
            Attack_Models[c][l] = DepthNN(nInputs)
            # TODO create one nn per depth
    return Attack_Models

def experiment1():

    logger = MyLogger(Config(experimentName = "experiment 1",
                            nameTest= " test",
                            model="centralised",
                            dataset="healthcare",
                            lam=0.1, # 0.1 10
                            gamma=0.5,
                            alpha=0.0,
                            learning_rate=0.3,
                            max_depth=8,
                            max_tree=20,
                            nBuckets=35,
                            save=False)).logger  # hacky way to get A logger

    seed = 10
    datasets = ["synthetic-10", "synthetic-20", "synthetic-50", "synthetic-100"]

    targetArchitectures = ["XGBoost", "FederBoost-central"]
    targetArchitectures = ["FederBoost-central"]

    all_data = {} # all_data["XGBoost"]["healthcore"]["accuarcy train shadow"]
    # tested_param = "gamma"
    # tested_param_vals = [0.1, 0.5, 0.7, 0.9, 1]
    to_be_tested = {"gamma": [0, 0.1, 0.25, 0.5, 0.75, 1, 5, 10], # [0,inf] minimum loss for split to happen default = 0
                    "max_depth": [5, 8, 12, 15],
                    "max_tree": [5, 10, 20, 30, 50, 100, 150],
                    # "training_size": [1000, 2000, 5000, 10_000, 30_000],
                    "alpha": [0, 0.1, 0.25, 0.5, 0.75, 1, 10],  # [0, inf] L1 regularisation default = 0
                    "lam":   [0, 0.1, 0.25, 0.5, 0.75, 1, 10],  # L2 regularisation [0, inf] default = 1
                    "learning_rate":   [0.1, 0.25, 0.5 ,0.75, 1]  # learning rate [0,1] default = 0.3
                    }

    # to_be_tested = {"gamma": [0, 0.1],
    #                 "max_depth": [5, 10]}

    all_data = {} # all_data["XGBoost"][tested_metic]["healthcare"]["accuarcy train shadow"]
    # plot_auc(y_tre, y_pred, "Plots/experiment1_AUC.jpeg")
    
    for targetArchitecture in targetArchitectures:
        all_data[targetArchitecture] = {}
        for parameter_name, params in to_be_tested.items():
            all_data[targetArchitecture][parameter_name] = {}
            for dataset in datasets:
                all_data[targetArchitecture][parameter_name][dataset] = {}
            
                for val in params:
                    config = Config(experimentName = "experiment 1",
                            nameTest= dataset + " test",
                            model="normal",
                            dataset=dataset,
                            lam=0.1, # 0.1 10
                            gamma=0.5,
                            alpha=0.0,
                            learning_rate=0.3,
                            max_depth=8,
                            max_tree=20,
                            nBuckets=35,
                            save=False)
                    create_table_config(config.alpha, config.gamma, config.lam, config.learning_rate, config.max_depth, config.max_depth, 
                                        "experiment1_targetandshadow")
                    
                    setattr(config, parameter_name, val)  # update the config with the to be tested value
                    print(f"metric {parameter_name} {val}")
                    logger.warning(f"metric {parameter_name} {val}")
                    target_model = SFXGBoost(config, logger)
                    shadow_model = SFXGBoost(config, logger)
                    # attack_model = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
                    attack_model = xgb.XGBClassifier(max_depth=12, objective="binary:logistic", tree_method="approx",
                                    learning_rate=0.3, n_estimators=20, gamma=0.5, reg_alpha=0.0, reg_lambda=0.1)
                    data = get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger)
                    if rank == PARTY_ID.SERVER:
                        all_data[targetArchitecture][parameter_name][dataset][val] = data
                        
    if rank == PARTY_ID.SERVER:

        name_model = targetArchitectures[0]
        metrics = ["accuracy test attack", "precision test attack", "recall test attack", "overfitting target"]
        arguments = (all_data, to_be_tested, metrics, name_model, datasets)
        pickle.dump(arguments, open("./Saves/arguments.p", 'wb'))
        create_latex_table_1(all_data, to_be_tested, metrics, name_model, datasets, "./Table/experiment_1_.txt")

def get_standard_config_experiement2(dataset):
    config = Config(experimentName = "experiment 2",
            nameTest= dataset + " test",
            model="normal",
            dataset=dataset,
            lam=0.1, # 0.1 10
            gamma=0.5,
            alpha=0.0,
            learning_rate=0.3,
            max_depth=8,
            max_tree=20,
            nBuckets=35,
            save=True,
            target_rank=1)
    return config 

def experiment2dot1():
    datasets = ["healthcare"]
    target_Architectures = ["FederBoost-federated"]

    all_data={}
    for target_Architecture in target_Architectures:
        all_data[target_Architecture] = {}
        for dataset in datasets:
            pass

def experiment2():
    seed = 10
    # datasets = ["healthcare", "MNIST", "synthetic", "Census", "DNA", "Purchase-10", "Purhase-20", "Purchase-50", "Purchase-100"]
    # datasets = ["synthetic"]
    # datasets = ["healthcare", "synthetic", "purchase-10", "purchase-20", "purchase-50", "purchase-100", "texas"]

    # datasets = ["healthcare", "synthetic-10", "synthetic-20", "synthetic-50", "synthetic-100"]
    # datasets = ["synthetic-10", "synthetic-100"]
    datasets = ["healthcare"]


    targetArchitectures = ["XGBoost", "FederBoost-central", "FederBoost-federated"]
    # targetArchitectures = ["FederBoost-federated"]

    all_data = {} # all_data["XGBoost"]["healthcore"]["accuarcy train shadow"]
    for targetArchitecture in targetArchitectures:
        all_data[targetArchitecture] = {}
        for dataset in datasets:
            config = Config(experimentName = "experiment 2",
            nameTest= dataset + " test",
            model="normal",
            dataset=dataset,
            lam=0.1, # 0.1 10
            gamma=0,
            alpha=0.0,
            learning_rate=0.3,
            max_depth=12,
            max_tree=20,
            nBuckets=35,
            save=True,
            target_rank=0)
            logger = MyLogger(config).logger
            if rank == PARTY_ID.SERVER: logger.warning(config.prettyprint())
            create_table_config(config.alpha, config.gamma, config.lam, config. learning_rate, config.max_depth, config.max_tree, 
                                "experiment 2 targetandshadow")
            np.random.RandomState(seed) # TODO set seed sklearn.split
            
            if targetArchitecture == "FederBoost-central":
                target_model = SFXGBoost(config, logger)
                shadow_model = SFXGBoost(config, logger)
                # attack_model = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
                attack_model = xgb.XGBClassifier(max_depth=12, objective="binary:logistic", tree_method="approx",
                                learning_rate=0.3, n_estimators=20, gamma=0.5, reg_alpha=0.0, reg_lambda=0.1)
                data = get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger, targetArchitecture)
                if rank == PARTY_ID.SERVER:
                    all_data[targetArchitecture][dataset] = data
            elif targetArchitecture == "FederBoost-federated":
                target_model = SFXGBoost(config, logger)

                shadow_models = [SFXGBoost(config, logger) for _ in range(10)]  # TODO create the amount of shadow models allowed given by datasetretrieval
                
                attack_models1 = [ [MLPClassifier(hidden_layer_sizes=(300, 100,50,1), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000) for l in range(config.max_depth)] for c in range(config.nClasses)]
                # create_table_config(alpha)
                # attack_models1 = [ [xgb.XGBClassifier(max_depth=12, objective="binary:logistic", tree_method="approx",
                #         learning_rate=0.3, n_estimators=20, gamma=0.5, reg_alpha=0.5, reg_lambda=0.1) for l in range(config.max_depth)] for c in range(config.nClasses)]
                
                attack_model2 = xgb.XGBClassifier(max_depth=12, objective="binary:logistic", tree_method="exact",
                    learning_rate=0.3, n_estimators=20, gamma=0.5, reg_alpha=0.5, reg_lambda=0.1)
                
                data = train_all_federated(target_model, shadow_models, attack_models1, attack_model2, config, logger)
                if rank == PARTY_ID.SERVER:
                    all_data[targetArchitecture][dataset] = data
            elif targetArchitecture == "XGBoost" and rank == PARTY_ID.SERVER:  # define neural network
                target_model = xgb.XGBClassifier(max_depth=config.max_depth, objective="multi:softmax", tree_method="approx", num_class=config.nClasses,
                        learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
                shadow_model = xgb.XGBClassifier(max_depth=config.max_depth, objective="multi:softmax", tree_method="approx", num_class=config.nClasses,
                        learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
                # attack_model = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
                attack_model = xgb.XGBClassifier(max_depth=12, objective="binary:logistic", tree_method="approx",
                            learning_rate=0.3, n_estimators=20, gamma=0.5, reg_alpha=0.0, reg_lambda=0.1)
                data = get_data_attack_centrally(target_model, shadow_model, attack_model, config, logger, targetArchitecture)
                all_data[targetArchitecture][dataset] = data
            else:
                continue
                # raise Exception("Wrong model types given!")
            
            # saver.save_experiment_2_one_run(attack_model, )
    if rank == PARTY_ID.SERVER:
        save(all_data, name="all_data federated", config=config)
    plot_experiment2(all_data)
        
        # saver.create_plots_experiment_2(all_data)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <exeperiment number>")
    else:
        experimentNumber = int(sys.argv[1])
        print(f"running experiment {experimentNumber}!")
        if experimentNumber == 1:
            experiment1()
        elif experimentNumber == 2:
            experiment2()
    if False:
        dataset = "healthcare"
        config = Config(experimentName="tmp",
            nameTest= dataset + "klein 20 trees",
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
            # shadow_model = xgb.XGBClassifier(max_depth=config.max_depth, objective="multi:softmax", tree_method="approx",
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

        X, y, X_test, y_test, y_pred_org, target_model, X_shadow, y_shadow, fName = test_global(config, logger, target_model, getDataBase(config.dataset, POSSIBLE_PATHS))
        
        preform_attack_centralised(config, logger, (X_shadow, y_shadow), target_model, shadow_model, attack_model, X, y, X_test, y_test, fName)

        if rank == 0:
            from sklearn.metrics import accuracy_score
            print(f"y_test = {np.argmax(y_test, axis=1)}")
            print(f"y_pred_org = {y_pred_org}")
            print(f"Accuracy federboost = {accuracy_score(np.argmax(y_test, axis=1), y_pred_org)}")

# import cProfile
# cProfile.run('main()', sort='cumtime')
main()
