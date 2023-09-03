from SFXGBoost.common.pickler import retrieve
import pickle

# shadow = pickle.load(open('Saves/healthcare test_rank_0/target_model.p', 'rb'))
# shadow = pickle.load(open('Saves/healthcare test_/rank_0/shadow_model_0.p', 'rb'))
# shadow = pickle.load(open('Saves/healthcare test_rank_0/all_data federated.p', 'rb'))

x = 1
pass

from SFXGBoost.view.plotter import plot_histogram


datasets = ("Healthcare", "MNIST")
data = {"XGBoost":[0.5, 0.6],
        "FederBoost Centralised":(0.5, 0.6),
        "FederBoost Federated":(0.6, 0.8)}
plot_histogram(datasets, data, "Plots/test.jpeg")
pass