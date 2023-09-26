import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
from SFXGBoost.view.plotter import plot_auc
from SFXGBoost.config import Config
import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

D_train = pickle.load(open('/mnt/scratch_dir/meerhofj/Saves/healthcare test_rank_0/healthcare_D_Train_Attack_0,7.p', 'rb'))

import xgboost as xgb
attack_model_1 = xgb.XGBClassifier(max_depth=12, objective="binary:logistic", tree_method="approx",
                        learning_rate=0.3, n_estimators=30, gamma=0, reg_alpha=0, reg_lambda=1)

attack_model_1 = attack_model_1.fit(D_train[0], D_train[1])
from SFXGBoost.MemberShip import get_train_Attackmodel_1
# D_test = get_train_Attackmodel_1(config, None, target_model, 0, 7, D_train)
attack_model_1.predict()