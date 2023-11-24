
# creates plots 
import matplotlib.pyplot as plt
import matplotlib.axis as Axis
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from SFXGBoost.config import Config

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
    plot_histogram(datasets, data_acc, title="accuracy attack", y_label="accuracy", destination=f"Plots/experiment{experiment_number}_acc_attack.pdf", plot_schema=False, red_line=True)
    plot_histogram(datasets, data_prec, title="precision attack", y_label="precision", destination=f"Plots/experiment{experiment_number}_precision.pdf", plot_schema=False, red_line=True)
    plot_histogram(datasets, data_acc_test, title="accuracy test target", y_label="accuracy", destination=f"Plots/experiment{experiment_number}_acc_test.pdf", plot_schema=False)
    plot_histogram(datasets, data_overfitting, title="overfitting target", y_label="overfitting", destination=f"Plots/experiment{experiment_number}_overfitting.pdf", plot_schema=False)

def plot_loss(Train_loss, test_loss, config:Config):
    from datetime import date
    import time
    print(f"Train Loss = {Train_loss}")
    print(f"Test loss = {test_loss}")
    print(f"config info: {config.dataset}, {config.targetArchitecture}")
    day = date.today().strftime("%b-%d-%Y")
    curTime = time.strftime("%H:%M", time.localtime())
    destination = f"Plots/experimentx-Loss{config.dataset},{config.targetArchitecture},{day},{curTime}.pdf"
    trees = range(1, len(Train_loss)+1)
    fig, ax = plt.subplots(layout='constrained')
    
    ax.plot(trees, Train_loss, label="Training Loss", marker='o')
    ax.plot(trees, test_loss, label="Testing Loss", marker='*')
    plt.xticks(trees)
    plt.xlabel('Tree')
    plt.ylabel('Loss')
    plt.title("Loss over Trees")
    ax.legend()
    
    fig.savefig(destination, dpi=1200, format='pdf')


def plot_histogram(datasets, data, title="Sample text", y_label="y_label", destination="Plots/experiment2.pdf", plot_schema:bool=True, red_line:bool=False):
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
    if red_line:
        ax.axhline(0.5, color='red', linestyle='--', zorder=-10)
    legend_handles = []
    for attribute, measurements in data.items():
        measurements = [round(value, 3,) for value in measurements]
        if np.max(measurements) >= 0.9:
            one_value = True
        offset = width * multiplier
        rects = ax.bar(x+offset, measurements, width, label=attribute)
        ax.bar_label(rects, label_type='edge', padding=-15)
        multiplier += 1
        legend_handles.append(rects)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x+width, datasets)
    if plot_schema:
        ax.legend(loc='upper left', ncol=3)
    if False:
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylim(0, 1)
    plt.savefig(destination, dpi=1200, format='pdf')
    if not plot_schema:
        fig_legend, ax_legend = plt.subplots(figsize=(6,1))
        fig_legend.legend(handles=legend_handles, loc='center', ncol=3)
        # ax.get_legend()
        ax_legend.axis('off')
        destination = destination.replace(".pdf", f"_legend.pdf")
        fig_legend.savefig(destination, dpi=1200, format='pdf')

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
 
 # Last week loss plotting addition, I kinda just copied the loss numbers out of the output.

'''
# import pickle
# all_data = pickle.load(open("/home/meerhofj/Desktop/results/26-oct-overfitted/synthetic-100_all_data federated 3.p", "rb"))
# plot_experiment(all_data, 3)
TrainHealth = [2.011893834833509, 1.7091067470915868, 1.544512335182848, 1.4204669790203366, 1.3197647120105025, 1.2293683092876422, 1.1598061378586073, 1.0891434351038352, 1.0333679944794483, 0.9727745182290487, 0.9183800174940348, 0.8628151410616234, 0.8192126364364399, 0.7759235238432882, 0.7319552072274018, 0.6893938983695553, 0.6558652483663203, 0.626487358440389, 0.601995788022876, 0.5728022523738859]
TestHealth = [2.3114516249205725, 2.2267525099285224, 2.2115413121798237, 2.206956061351048, 2.2021813053585015, 2.2108302575944965, 2.215029610554474, 2.223430665658443, 2.2301375563784385, 2.2328415054198913, 2.2397961656170438, 2.239961787378641, 2.2478138010669273, 2.2508754318991033, 2.2527239079520958, 2.257149943964193, 2.2656874931962347, 2.272273351498682, 2.283279398578291, 2.289944956333592]
Trainsyn10 = [1.6748756732256274, 1.396677104472225, 1.2390018699620449, 1.1094097589221859, 1.009525526201071, 0.932168781268361, 0.8680923393670059, 0.8072374901940801, 0.7436733625978246, 0.6897565744568644, 0.645703245199575, 0.6075278787994395, 0.5630595173236971, 0.5246850607328987, 0.48758338464381146, 0.4637790849472613, 0.4372704167727629, 0.41390493223485725, 0.39450595905286545, 0.3730883350624726]
Testsyn10 = [1.9632191125192007, 1.8487543457332987, 1.797689811735166, 1.7520266494204206, 1.7166438328932472, 1.6931242073322847, 1.6752464020042945, 1.6524456259518523, 1.6277599798121074, 1.6090735922346704, 1.5933846540534096, 1.582934524972081, 1.557973188411649, 1.5427098575989537, 1.5279173739153868, 1.519493913243979, 1.510434784401278, 1.4992578440683888, 1.4902577299268742, 1.4790031820857232]
Trainsyn100 = [3.573036914872103, 2.777121211524608, 2.123185236327843, 1.6158029759985937, 1.2558737750585869, 0.9961502358529735, 0.8083525330417571, 0.6608073915311201, 0.5464757371799118, 0.4534211808750232, 0.3810530976148765, 0.3218688744043515, 0.27229222385513036, 0.23297473390536091, 0.201621099552596, 0.17680245995922353, 0.15734318669246933, 0.14141579046071734, 0.1276712962144349, 0.11675075389155234]
Testsyn100 = [4.385449562819065, 4.262290956298024, 4.183960302101514, 4.140470538645909, 4.116427904892592, 4.08991201297353, 4.07874250160127, 4.071705387862553, 4.0669944107897695, 4.055937883904822, 4.049742925795262, 4.042667512696818, 4.043732720980665, 4.042940784294965, 4.040518360700204, 4.045237961688347, 4.051762649935084, 4.051834927634715, 4.0553830843361816, 4.058397688700761]

from datetime import date
import time
day = date.today().strftime("%b-%d-%Y")
curTime = time.strftime("%H:%M", time.localtime())
destination = f"Plots/experimentx-Losstmp,{day},{curTime}.pdf"
trees = range(1, 20+1)
fig, ax = plt.subplots(layout='constrained')

ax.plot(trees, TrainHealth, color='m', label="Healthcare", marker='d')
ax.plot(trees, TestHealth, color='m', label="Testing Loss", marker='*')
ax.plot(trees, Trainsyn10, color='r', label="Synthetic-10", marker='d')
ax.plot(trees, Testsyn10, color='r', label="Testing Loss", marker='*')
ax.plot(trees, Trainsyn100, color='gray', label="Synthetic-100", marker='d')
ax.plot(trees, Testsyn100, color='gray', label="Testing Loss", marker='*')
legend_marker1 = plt.Line2D([0], [0], marker='d', color='black', markerfacecolor='black', markersize=10, label='Training Loss')
legend_marker2 = plt.Line2D([0], [0], marker='*', color='black', markerfacecolor='black', markersize=10, label='Testing Loss')
health = plt.Line2D([0], [0], color='m', label='healthcare')
syn10 = plt.Line2D([0], [0], color='r', label='synthetic-10')
syn100 = plt.Line2D([0], [0], color='gray', label='synthetic-100')



plt.xticks(trees)
plt.xlabel('Tree')
plt.ylabel('Loss')
plt.title("Loss over Trees")
ax.legend(handles=[health, syn10, syn100, legend_marker1, legend_marker2], loc='best')

fig.savefig(destination, dpi=1200, format='pdf')
'''