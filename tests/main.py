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
def train_all(target_model, shadow_model_s, attack_model, config, logger):
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS)()
    log_distribution(logger, X_train, y_train, y_test)

    target_model.fit(X_train, y_train, fName)
    D_Train_Shadow, D_Out_Shadow, D_Test = split_shadow((X_shadow, y_shadow))
    if type(shadow_model_s) == SFXGBoost:
        shadow_model_s.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName)
    else:
        for shadow_model in shadow_model_s:
            shadow_model.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName)  # TODO split shadow model such that they are not all the same!

    z, labels = create_D_attack_centralised(shadow_model_s, D_Train_Shadow, D_Out_Shadow)

    attack_model.fit(z, labels)

def experiment2():
    seed = 10

    datasets = ["healtcare", "MNIST", "synthetic", "Census", "DNA", "Purchase-10", "Purhase-20", "Purchase-50", "Purchase-100"]
    targetArchitectures = ["XGBoost", "FederBoost", ]
    for targetArchitecture in targetArchitectures:

        for dataset in datasets:
            config = Config(experimentName= "experiment 2",
            nameTest= dataset + " test",
            model="normal",
            dataset=dataset,
            lam=0.1, # 0.1 10
            gamma=0.5,
            alpha=0.0,
            learning_rate=1,
            max_depth=5,
            max_tree=9,
            nBuckets=100)
            logger = MyLogger(config).logger
            logger.warning(config.prettyprint())
            np.random.RandomState(seed) # TODO set seed sklearn.split
            
            if targetArchitecture == "Federboost":
                target_model = SFXGBoost(config, logger)
                shadow_model = SFXGBoost(config, logger)
                shadow_models = [SFXGBoost(config, logger) for _ in range(10)]  # 
                
                attack_model_central = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
                attack_model_federated = None  # TODO define federated neural network.


            elif targetArchitecture == "XGBoost":  # define neural network
                target_model = xgb.XGBClassifier(ax_depth=config.max_depth, objective="multi:softmax", tree_method="approx",
                        learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
                shadow_model = xgb.XGBClassifier(ax_depth=config.max_depth, objective="multi:softmax", tree_method="approx",
                        learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
                attack_model = MLPClassifier(hidden_layer_sizes=(20,11,11), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=2000)
            else:
                raise Exception("Wrong model types given!")
            
            train_all(target_model, shadow_model, attack_model, config, logger)
            saver.save_experiment_2_one_run(attack_model, )

    saver.create_plots_experiment_2


def main():
    dataset = "texas"
    config = Config(nameTest= dataset + " test",
           model="normal",
           dataset=dataset,
           lam=0.1, # 0.1 10
           gamma=0.5,
           alpha=0.0,
           learning_rate=1,
           max_depth=5,
           max_tree=9,
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
    
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS)()
    
    log_distribution(logger, X_train, y_train, y_test)
    target_model.fit(X_train, y_train, fName)
    D_Train_Shadow, D_Out_Shadow, D_Test = split_shadow((X_shadow, y_shadow))
    shadow_model.fit(D_Train_Shadow[0], D_Train_Shadow[1], fName)

    z, labels = create_D_attack_centralised(shadow_model, D_Train_Shadow, D_Out_Shadow)

    attack_model.fit(z, labels)

    save_metrics_one_run(target_model, shadow_model, attack_model, (X_train, y_train), (X_test, y_test), D_Train_Shadow, (z, labels), D_Test)

    X, y, y_pred_org, y_test, target_model, X_shadow, y_shadow, fName = test_global(config, logger, target_model, getDataBase(config.dataset, POSSIBLE_PATHS))
    
    preform_attack_centralised(config, (X_shadow, y_shadow), target_model, shadow_model, attack_model, X, y, fName)

    if rank == 0:
        from sklearn.metrics import accuracy_score
        print(f"y_test = {np.argmax(y_test, axis=1)}")
        print(f"y_pred_org = {y_pred_org}")
        
        print(f"Accuracy federboost = {accuracy_score(np.argmax(y_test, axis=1), y_pred_org)}")

main()