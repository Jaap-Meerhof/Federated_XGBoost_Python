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

# 10,5 works really well. 
D_train = pickle.load(open('Saves/healthcare test_rank_0/D_Train_Attack_7,7.p', 'rb'))
x = 1
# D_train = pickle.load(open('Saves/healthcare test_rank_0/D_train_attack2.p', 'rb'))
x=1




attack_model_2 = MLPClassifier(hidden_layer_sizes=(1000, 100), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000)
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
            max_tree=20,
            nBuckets=35)

attack_model_2 = xgb.XGBClassifier(max_depth=config.max_depth, objective="binary:logistic", tree_method="approx",
                        learning_rate=config.learning_rate, n_estimators=config.max_tree, gamma=config.gamma, reg_alpha=config.alpha, reg_lambda=config.lam)
# attack_model_2 = MLPClassifier(hidden_layer_sizes=(1000,100), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000)

def tmp(x_test):
    x_test2 = x_test[:, 0:16]
    x_test3 = x_test[:, 27:]
    x_test_1 = np.concatenate((x_test2, x_test3), axis=1)
    return x_test_1


x = D_train[0][0:30_000]
y = D_train[1][0:30_000]
x_test = D_train[0][30_000:40_000]
y_test = D_train[1][30_000:40_000]

x_new = tmp(x)
# x_new = x
# x_new = x[:, 16:27]

x_test_2 = tmp(x_test)
# x_test_2 = x_test
# x_test_2 = x_test[:, 16:27]

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(x_new.shape[1],)),
    keras.layers.Dense(64, activation='relu', input_shape=(x_new.shape[1],)),

    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_new, y, epochs=5, batch_size=16)
attack_model_2.fit(x_new, y)




y_pred_proba = attack_model_2.predict_proba(x_test_2)
y_pred_proba_nn = model.predict(x_test_2)

y_pred = attack_model_2.predict(x_test_2)

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
