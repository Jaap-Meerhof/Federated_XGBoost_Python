from SFXGBoost.common.pickler import retrieve
import pickle

# shadow = pickle.load(open('Saves/healthcare test_rank_0/target_model.p', 'rb'))
shadow = pickle.load(open('Saves/healthcare test_rank_1/shadow_model_0.p', 'rb'))
# all_data = pickle.load(open('Saves/healthcare test_rank_0/all_data federated.p', 'rb'))
all_data = pickle.load(open('Saves/all_data federated_backup.p', 'rb'))

# D_train = pickle.load(open('Saves/healthcare test_rank_0/D_train_attack2.p', 'rb'))

from SFXGBoost.view.plotter import plot_histogram, plot_experiment2
plot_experiment2(all_data)
x = 1
pass



# datasets = ("Healthcare", "MNIST")
# data = {"XGBoost":[0.5, 0.6],
#         "FederBoost Centralised":(0.5, 0.6),
#         "FederBoost Federated":(0.6, 0.8)}
# plot_histogram(datasets, data, "Plots/test.jpeg")
# pass

