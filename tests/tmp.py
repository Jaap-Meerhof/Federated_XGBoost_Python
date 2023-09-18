from SFXGBoost.common.pickler import retrieve
import pickle
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Assuming y_true contains true labels (0 or 1) and y_pred contains predicted probabilities
y_true = [0,0,0,0,0,1,1,1,0,1]
y_pred = [0.001, 0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9,0.9999]

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

best_threshold_index = np.argmax(tpr - fpr)
best_threshold = thresholds[best_threshold_index]

# Calculate AUC
auc = roc_auc_score(y_true, y_pred)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], c='red', marker='o', label=f'Best Threshold = {best_threshold:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()






# shadow = pickle.load(open('Saves/healthcare test_rank_0/target_model.p', 'rb'))
shadow = pickle.load(open('Saves/healthcare test_rank_0/shadow_model_0.p', 'rb'))
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

