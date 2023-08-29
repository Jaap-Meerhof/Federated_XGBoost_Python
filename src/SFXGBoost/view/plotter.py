
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

def plot_histogram(datasets, data, destination="Results/experiment2.jpeg"):
    

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
    plt.show()
    plt.savefig(destination, format="jpeg", dpi=1200, bbox_inches='tight')

