
# creates plots 
import matplotlib.pyplot as plt
import matplotlib.axis as Axis
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_acc_one(data, labels, destination='plot.png', name='Sample Text', subtext='Sample Text sub'):
    """
    # labels = "tested_param_name (like "gamma"), attack-centralised", "optional", ...
    # data = (, len(labels))

    Args:
        data (_type_): _description_
        labels (_type_): _description_
        destination (str, optional): _description_. Defaults to 'plot.png'.
        name (str, optional): _description_. Defaults to 'Sample Text'.
        subtext (str, optional): _description_. Defaults to 'Sample Text sub'.
    """
    colorlist = [ "darkred", "darkkhaki", "red", "b", "g", "orange", "magenta", "black", "lime"]
    markerlist = ['*',  '|', 'o', 's', 'v', 'd', 'x', 'h']
    
    for i, row in enumerate(data, start=1):
        plt.plot(data[:, i], data[:, 0], color=colorlist[i], marker=markerlist[i], label=labels[i])
    plt.xlabel = data[0,0]
    plt.ylabel = "accuracy"
    plt.title = name
    plt.legend(loc='best')
    plt.grid(True)
    # plt.show()
    plt.savefig(destination, format="pdf", dpi=1200, bbox_inches='tight')

def plot_experiment(all_data:dict, experiment_number:int):
    """Plots accuracy and precision of the attack in one plot with two subplots
    and plots accuracy and precision of target model in one plot

    All_data = {name_network: {dataset:{precision,...,metric}}}
    Args:
        all_data (_type_): _description_
        destination (str, optional): _description_. Defaults to f"Plots/experiment{experiment_number}.pdf".
    """
    data_acc = {}
    data_prec = {}
    data_acc_test = {}
    data_overfitting = {}

    for name, datasets in all_data.items():
        data_acc[name] = []
        data_prec[name] = []
        data_acc_test[name] = []
        data_overfitting[name] = []
        for dataset, metrics in datasets.items():
            precision_attack = metrics["precision test attack"]
            accuracy_attack = metrics["accuracy test attack"]
            accuracy_test = metrics["accuracy test target"]
            overfitting = metrics["overfitting target"]
            data_acc[name].append(accuracy_attack)
            data_prec[name].append(precision_attack)
            data_acc_test[name].append(accuracy_test)
            data_overfitting[name].append(overfitting)
        
    datasets = list(all_data[list(all_data.keys())[0]].keys())
    plot_histogram(datasets, data_acc, title="accuracy attack", y_label="accuracy", destination=f"Plots/experiment{experiment_number}_acc_attack.pdf")
    plot_histogram(datasets, data_prec, title="precision attack", y_label="precision", destination=f"Plots/experiment{experiment_number}_precision.pdf")
    plot_histogram(datasets, data_acc_test, title="accuracy test target", y_label="accuracy", destination=f"Plots/experiment{experiment_number}_acc_test.pdf")
    plot_histogram(datasets, data_overfitting, title="overfitting target", y_label="overfitting", destination=f"Plots/experiment{experiment_number}_overfitting.pdf")



def plot_histogram(datasets, data, title="Sample text", y_label="y_label", destination="Plots/experiment2.pdf"):
    """_summary_
    example
    datasets = ("Healthcare", "MNIST")
    data = {"XGBoost":(0.5, 0.6),
        "FederBoost Centralised":(0.5, 0.6),
        "FederBoost Federated":(0.6, 0.8)}

    Args:
        datasets (_type_): _description_
        data (_type_): _description_
        destination (str, optional): _description_. Defaults to "Plots/experiment2.pdf".
    """
    from datetime import date
    import time
    day = date.today().strftime("%b-%d-%Y")
    curTime = time.strftime("%H:%M", time.localtime())
    destination = destination.replace(".pdf", f"{day},{curTime}.pdf")
    width = 0.25
    multiplier = 0
    one_value = False
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurements in data.items():
        measurements = [round(value, 3,) for value in measurements]
        if np.max(measurements) >= 0.9:
            one_value = True
        offset = width * multiplier
        rects = ax.bar(x+offset, measurements, width, label=attribute)
        ax.bar_label(rects, label_type='edge', padding=-15)
        multiplier += 1
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x+width, datasets)
    ax.legend(loc='upper left', ncol=3)
    if one_value:
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylim(0, 1)
    plt.savefig(destination, dpi=1200, format='pdf')
    # plt.show()

def plot_auc(y_true, y_pred, destination="Plots/experiment2_AUC_attack2.pdf"):
    """plots a AUC curve that will also display the best threshold in the legend. 
    thanks chatGPT for creating this 

    Args:
        y_true (np.array): Binary true values classifications 1 or 0
        y_pred (np.array): binary probability scores from 0.0 to 1.0
    """
    from datetime import date
    import time
    day = date.today().strftime("%b-%d-%Y")
    curTime = time.strftime("%H:%M", time.localtime())
    destination = destination.replace(".pdf", f"{day},{curTime}.pdf")

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
    plt.savefig(destination, dpi=1200, format='pdf')