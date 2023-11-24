from SFXGBoost.view.plotter import plot_auc, plot_experiment
import pickle

arguments = pickle.load(open("./Saves/arguments_backup.p", "rb"))
shadow = pickle.load(open("delme.p", "rb"))
from SFXGBoost.data_structure.treestructure import FLTreeNode
c=0
d=7
p=1
node = shadow.nodes[c][d][4]
w, scorep = FLTreeNode.compute_leaf_param(node.Gpi[p][1], node.Hpi[p][1], 0.1, 0.5)
all_data = pickle.load(open("/mnt/scratch_dir/meerhofj/Saves/experiment 3/high overfitting 1_rank_0/synthetic-100_all_data federated 3.p", "rb"))
plot_experiment(all_data, 3)
x = 1


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
from SFXGBoost.config import Config
import numpy as np

import numpy as np
# import tensorflow as tf
# from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from SFXGBoost.common.pickler import save
from SFXGBoost.view.table import create_latex_table_1
# var = 9
# gpi = pickle.load(open("Gpi.p", "rb"))


# create_latex_table_1(all_data=arguments[0], to_be_tested=arguments[1], metrics=arguments[2], name_model=arguments[3], datasets=arguments[4], destination="./Table/experiment_1.txt")
# 10,5 works really well. 
shadow = pickle.load(open('/mnt/scratch_dir/meerhofj/Saves/healthcare test_rank_0/healthcare_shadow_model_1.p', 'rb'))
D_train = pickle.load(open('/mnt/scratch_dir/meerhofj/Saves/healthcare test_rank_0/healthcare_D_Train_Attack_0,3.p', 'rb'))
all_data = pickle.load(open('/mnt/scratch_dir/meerhofj/Saves/healthcare test_rank_0/healthcare_all_data federated.p', 'rb'))


# D_test = pickle.load(open('Saves/healthcare test_rank_0/healthcare_D_test_attack2.p', 'rb'))

# tmp = pickle.load(open('Saves/healthcare test_rank_0/healthcare_D_train_attack2.p', 'rb'))
if False:
    x = 1
    D_train = pickle.load(open('/mnt/scratch_dir/meerhofj/Saves/healthcare test_rank_0/healthcare_D_train_attack2.p', 'rb'))
    np.column_stack((D_train[0][7], D_train[0][14], D_train[0][21]))
    D_test = pickle.load(open('/mnt/scratch_dir/meerhofj/Saves/healthcare test_rank_0/healthcare_D_test_attack2.p', 'rb'))
    tmp = np.average(D_train[0], axis=1)
    x=1

    ####################
    import xgboost as xgb
    config = Config(experimentName = "experiment 2",
                nameTest= " test",
                model="normal",
                dataset="healthcare",
                lam=0.1, # 0.1 10
                gamma=0.5,
                alpha=0.2,
                learning_rate=0.3,
                max_depth=15,
                max_tree=30,
                nBuckets=35)

    attack_model_2 = xgb.XGBClassifier(max_depth=config.max_depth, objective="binary:logistic", tree_method="approx",
                            learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
    attack_model_2.fit(D_train[0], D_train[1])
    y_pred = attack_model_2.predict(D_test[0])
    acc = accuracy_score(D_test[1], y_pred)
    print(acc)
    x= 1
##########


# attack_model_2 = MLPClassifier(hidden_layer_sizes=(1000, 100), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000)
import xgboost as xgb
config = Config(experimentName = "experiment 2",
            nameTest= " test",
            model="normal",
            dataset="healthcare",
            lam=0.1, # 0.1 10
            gamma=0.5,
            alpha=0.2,
            learning_rate=0.3,
            max_depth=15,
            max_tree=30,
            nBuckets=35)
D_train = pickle.load(open('/mnt/scratch_dir/meerhofj/Saves/healthcare test_rank_0/healthcare_D_Train_Attack_2,2.p', 'rb'))
D_test = pickle.load(open('/mnt/scratch_dir/meerhofj/Saves/healthcare test_rank_0/healthcare_D_Test_Attack_2,2.p', 'rb'))

attack_model_2 = xgb.XGBClassifier(max_depth=config.max_depth, objective="binary:logistic", tree_method="approx",
                       learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
# attack_model_2 = MLPClassifier(hidden_layer_sizes=(1000,100), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000)

def tmp(x_test, y =None):
    # x_test2 = x_test[:, 0:16]
    # x_test3 = x_test[:, 27:]
    # x_test_1 = np.concatenate((x_test2, x_test3), axis=1)
    # indices = np.lexsort(x_test[:, :16].T)
    # x_test = x_test[indices]
    # y= y[indices]
    return x_test, y

# new_x, new_y = tmp(D_train[0], D_train[1])
# D_train = (new_x, new_y)
# if len(D_train[0]) < 29460:
#     raise Exception("D_train to small")
x = D_train[0][0:40_000]
y = D_train[1][0:40_000]
x_test = D_test[0][0:40_000]
y_test = D_test[1][0:40_000]
print(len(x))
# x_new = x
# x_new = x[:, 16:27]

# x_test_2 = x_test
# x_test_2 = x_test[:, 16:27]
from keras.layers import LeakyReLU
model = keras.Sequential([
    keras.layers.Dense(x.shape[1], activation=LeakyReLU(alpha=0.01), input_shape=(x.shape[1],)),
    keras.layers.Dense(500, activation=LeakyReLU(alpha=0.01), input_shape=(500,)),
    keras.layers.Dense(64, activation=LeakyReLU(alpha=0.01), input_shape=(300,)),
    keras.layers.Dense(1, activation='sigmoid')
])
optimiser = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=10, batch_size=16)
attack_model_2.fit(x, y)
import matplotlib.pyplot as plt
#from xgboost import plot_tree
#plot_tree(attack_model_2)
#plt.show()

y_pred_proba = attack_model_2.predict_proba(x_test)
y_pred_proba_nn = model.predict(x_test)

y_pred = attack_model_2.predict(x_test)

tmp = y_pred_proba[:,0]
plot_auc(y_test, y_pred_proba[:,0], "Plots/testing1.jpeg")
plot_auc(y_test, y_pred_proba_nn[:,0], "Plots/testing2.jpeg")
print(f"Accuracy xgboost normal = {accuracy_score(y_test, y_pred )}")
print(f"Accuracy xgboost normal nn = {accuracy_score(y_test, (y_pred_proba_nn > 0.5).astype(int) )}")
print(f"{(np.count_nonzero(y_test)*100) / y_test.shape[0]}")

# x = x[:, 16:27]
# attack_model_2 = xgb.XGBClassifier(ax_depth=config.max_depth, objective="binary:logistic", tree_method="approx",
#                         learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
# # attack_model_2 = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=1000)

# attack_model_2.fit(x, y)

# x_test = x_test[:, 16:27]
# y_pred_proba = attack_model_2.predict_proba(x_test)
# y_pred = attack_model_2.predict(x_test)

# plot_auc(y_test, y_pred_proba[:,1], "Plots/testing2.jpeg")
# print(f"Accuracy xgboost normal = {accuracy_score(y_test, y_pred )}")
