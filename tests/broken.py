
from sklearn.metrics import accuracy_score
import xgboost as xgb
from SFXGBoost.Model import SFXGBoost
from SFXGBoost.config import Config, MyLogger, rank
from SFXGBoost.Model import PARTY_ID
from SFXGBoost.dataset.datasetRetrieval import getDataBase
import numpy as np
# from tests.main import get_data_attack_centrally


datasets = ["healthcare"]
dataset = datasets[0]
config = Config(experimentName = "broken",
            nameTest= dataset + " broken",
            model="normal",
            dataset=dataset,
            lam=0.1, # 0.1 10
            gamma=0,
            alpha=0.0,
            learning_rate=0.3,
            max_depth=12,
            max_tree=20,
            nBuckets=35,
            save=False,
            target_rank=1)

logger = MyLogger(config).logger
central_config_attack= {"max_depth":12, "objective":"binary:logistic", "tree_method=":"approx", 
                             "learning_rate":0.3, "n_estimators":20, "gamma":0, "reg_alpha":0.0, "reg_lambda":1}
target_model = SFXGBoost(config, logger)
target_modelxgb = xgb.XGBClassifier(max_depth=config.max_depth, objective="multi:softmax", tree_method="approx",
                             learning_rate=0.3, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]
X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS, False, config.train_size)()
y_test = np.argmax(y_test, axis=1)

target_modelxgb.fit(X_train, y_test)
target_model.fit(X_train, y_train, fName) 

y_pred = target_model.predict(X_test)
y_predxgb = target_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
accxgb = accuracy_score(y_test, y_predxgb)
if rank == PARTY_ID.SERVER:
    print(f"acc= {acc}")
    print(f"accxgb = {accxgb}")


