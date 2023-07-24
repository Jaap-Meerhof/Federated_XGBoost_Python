from SFXGBoost.config import Config, rank, comm, MyLogger
import numpy as np
from logging import Logger
from SFXGBoost.Model import PARTY_ID, SFXGBoostClassifierBase, SFXGBoost
from SFXGBoost.data_structure.databasestructure import QuantiledDataBase, DataBase

def log_distribution(logger, X_train, y_train, y_test):
    nTrain = len(y_train)
    nZeroTrain = np.count_nonzero(y_train == 0)
    rTrain = nZeroTrain/nTrain

    nTest = len(y_test)
    nZeroTest = np.count_nonzero(y_test == 0)
    rTest = nZeroTest / nTest
    logger.warning("DataDistribution, nTrain: %d, zeroRate: %f, nTest: %d, ratioTest: %f, nFeature: %d", 
    nTrain, rTrain, nTest, rTest, X_train.shape[1])

def test_global(config:Config, logger:Logger, model: SFXGBoostClassifierBase, getDatabaseFunc):
    X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDatabaseFunc()
    log_distribution(logger, X_train, y_train, y_test)
    
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
    else:
        quantile = quantile.splitupHorizontal(start_end[rank-1][0], start_end[rank-1][1])
        model.setData(quantile, fName, X_train_my, y_train_my)
    
    model.boost(initprobability)

    if rank == PARTY_ID.SERVER:
        y_pred = model.predict(X_test, fName, initprobability)
        
        import xgboost as xgb
        xgboostmodel = xgb.XGBClassifier(max_depth=3, objective="multi:softmax",
                            learning_rate=0.3, n_estimators=10, gamma=0.5, reg_alpha=1, reg_lambda=10)
        xgboostmodel.fit(X_train, np.argmax(y_train, axis=1))
        from sklearn.metrics import accuracy_score
        y_pred_xgb = xgboostmodel.predict(X_test)
        print(f"Accuracy xgboost normal = {accuracy_score(y_test, y_pred_xgb)}")
        print(y_pred)
    else:
        y_pred = [] # basically a none

    y_pred_org = y_pred.copy()
    X = X_train
    y = y_train 
    return X, y, y_pred_org, y_test, model, X_shadow, y_shadow

dataset_list = ['purchase-10', 'purchase-20', 'purchase-50', 'purchase-100', 'texas', 'MNIST', 'synthetic', 'Census', 'DNA']
POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]
def main():
    config = Config(nameTest="test",
           model="normal",
           dataset="purchase-10",
           lam=0.1,
           gamma=1,
           max_depth=4,
           max_tree=5,
           nClasses=10,
           nFeatures=600)
    logger = MyLogger(config).logger
    from SFXGBoost.dataset.datasetRetrieval import getDataBase
    if config.model == "normal":
        model = SFXGBoost(config, logger)
        
    test_global(config, logger, model, getDataBase(config.dataset, POSSIBLE_PATHS))

main()