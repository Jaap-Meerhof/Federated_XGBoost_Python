
# creates plots 
import matplotlib.pyplot as plt
import matplotlib.axis as Axis
import numpy as np

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
    plt.show()
    plt.savefig(destination, format="jpeg", dpi=1200, bbox_inches='tight')

def plot_experiment2(all_data:dict, destination="Plots/experiment2.jpeg"):
    """Plots accuracy and precision of the attack in one plot with two subplots
    and plots accuracy and precision of target model in one plot

    All_data = {name_network: {dataset:{precision,...,metric}}}
    Args:
        all_data (_type_): _description_
        destination (str, optional): _description_. Defaults to "Plots/experiment2.jpeg".
    """
    for name, datasets in all_data.items():
        for dataset, metrics in datasets.items():
            precision_attack = metrics["precision test attack"]
            accuracy_attack = metrics["accuracy test attack"]
            data[name].append(accuracy_attack)
            data[name].append()
    datasets = list(all_data[list(all_data.keys())[0]].keys())
    data = {}
    pass

def plot_histogram(datasets, data, destination="Plots/experiment2.jpeg"):
    """_summary_
    example
    datasets = ("Healthcare", "MNIST")
    data = {"XGBoost":(0.5, 0.6),
        "FederBoost Centralised":(0.5, 0.6),
        "FederBoost Federated":(0.6, 0.8)}

    Args:
        datasets (_type_): _description_
        data (_type_): _description_
        destination (str, optional): _description_. Defaults to "Plots/experiment2.jpeg".
    """
    width = 0.25
    multiplier = 0
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurements in data.items():
        offset = width * multiplier
        rects = ax.bar(x+offset, measurements, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Precision')
    ax.set_title("precision accross datasets")
    ax.set_xticks(x+width, datasets)
    ax.legend(loc='upper left', ncol=3)
    ax.set_ylim(0, 1)
    plt.savefig(destination, dpi=1200, format='jpeg')
    plt.show()

