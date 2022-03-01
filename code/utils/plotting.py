import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# plotting functions

def plot_hist(data, metric_index, title="Insert title", savefig=False):
    names = ["RandomBaseline", "TransE", "DistMult", "ComplEx"]
    values = [list(data["RandomBaseline"].values())[metric_index], 
              list(data["TransE"].values())[metric_index], 
              list(data["DistMult"].values())[metric_index], 
              list(data["ComplEx"].values())[metric_index]]
    plt.bar(names, values, color=['red', 'green', 'blue', 'orange'])
    plt.ylabel('Score')
    plt.xlabel('Model class')
    plt.title(title)
    if savefig:
        plt.savefig("scores_" + str(metric_index) + ".png")
        
def plot_hit_scores(data, title="Insert title", savefig=False):
    """
    Plots a hits@n histogram
    """
    labels = ['Hits@1','Hits@3','Hits@10']
    metrics_RandomBaseline = list(data["RandomBaseline"].values())[2:5]
    metrics_TransE = list(data["TransE"].values())[2:5]
    metrics_DistMult = list(data["DistMult"].values())[2:5]
    metrics_ComplEx = list(data["ComplEx"].values())[2:5]
    x = (np.arange(len(labels)))*3  # the label locations
    width = 0.35  # the width of the bars
    mpl.style.use("bmh")
    fig, ax = plt.subplots()
    bar_C11 = ax.bar(x - 3*(width), metrics_RandomBaseline, width, label='RandomBaseline', color = 'red')
    bar_C12 = ax.bar(x - 2*(width), metrics_TransE, width, label='TransE', color = 'green')
    bar_C21 = ax.bar(x - width, metrics_DistMult, width, label='DistMult', color = 'blue')
    bar_C22 = ax.bar(x, metrics_ComplEx, width, label='ComplEx', color = 'orange')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    if savefig:
        plt.savefig("hit_scores.png")