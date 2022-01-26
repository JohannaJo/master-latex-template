# # Model selection for knowledge graph embeddings
# 
# For hyperparameter optimisation random search is more optimal than grid search as the search space grows: *James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(Feb):281–305, 2012.*
# 
# While this approach is not optimal, it is a strong baseline agains other more advanced methods such as Baysian optimisation: *Lisha Li and Kevin Jamieson. Hyperband: a novel bandit-based approach to hyperparameter optimization. Journal of Machine Learning Research, 18:1–52, 2018.*


import tensorflow as tf
import numpy as np
import pickle
np.random.seed(0)
import matplotlib as mpl
import matplotlib.pyplot  as plt

from ampligraph.evaluation import train_test_split_no_unseen 
from ampligraph.latent_features import RandomBaseline, TransE, DistMult, ComplEx#, HolE, ConvE, ConvKB
from ampligraph.latent_features import save_model
from ampligraph.evaluation import select_best_model_ranking # , mr_score, mrr_score, hits_at_n_score, evaluate_performance
from ampligraph.latent_features import save_model#, restore_model

# import module from different directory
# TODO: fix this bad solution
import sys
path = "/Home/siv30/fak006/Notebooks/Ampligraph/kb_extension_and_mining"
sys.path.insert(1, path)
from candidate_generation import get_entities

# set variables
MAX_COMBINATIONS = 25

# ## Data retrieval
data = np.loadtxt("family_subset.txt", dtype = 'object')
entities_subset = get_entities(data, max_entities=1000, selection_method='random')

X_train, X_val_test = train_test_split_no_unseen(data, test_size=0.2, seed=0)
X_val, X_test = train_test_split_no_unseen(data, test_size=0.05, seed=0)


test_metrics = {}


# ## Random Baseline
# Random baseline requires no hyperparameter search as it assigns a pseudo-random score to triples.
model_class = RandomBaseline
param_grid = {
    "seed": 0
}
best_model, _, _, _, randomBaseline_mrr_test, _ = select_best_model_ranking(model_class, X_train, X_val, X_test,
                          param_grid,
                          max_combinations=1,
                          use_filter=True,
                          verbose=False,
                          early_stopping=False, entities_subset=entities_subset)


test_metrics["RandomBaseline"] = randomBaseline_mrr_test
save_model(best_model, './trained_models/RandomBaseline.pkl')
del best_model


# ## TransE
model_class = TransE
param_grid = {
    "batches_count": [50],
    "seed": 0,
     "epochs": [10, 50, 100],
     "k": [100, 200],
     "eta": [5, 10, 15],
     "loss": ["pairwise", "nll"],
     "loss_params": {
         "margin": [2]
     },
     "embedding_model_params": {
     },
     "regularizer": ["LP", None],
     "regularizer_params": {
         "p": [1, 3],
         "lambda": [1e-4, 1e-5]
     },
     "optimizer": ["adagrad", "adam"],
     "optimizer_params": {
         "lr": lambda: np.random.uniform(0.0001, 0.01)
     },
     "verbose": False
}

best_model, _, _, _, transE_mrr_test, _ = select_best_model_ranking(model_class, X_train, X_val, X_test,
                          param_grid,
                          max_combinations=MAX_COMBINATIONS,
                          use_filter=True,
                          verbose=False,
                          early_stopping=False, entities_subset=entities_subset)

save_model(best_model, './trained_models/TransE.pkl')
test_metrics["TransE"] = transE_mrr_test
del best_model


# ## Distmult
model_class = DistMult
param_grid = {
    "batches_count": [50],
    "seed": 0,
    "epochs": [10, 50, 100],
    "k": [100, 200],
    "eta": [5, 10, 15],
    "loss": ["pairwise", "nll"],
    "loss_params": {
        "margin": [2]
    },
    "embedding_model_params": {
        
    },
    "regularizer": ["LP", None],
    "regularizer_params": {
        "p": [1, 3],
        "lambda": [1e-4, 1e-5]
    },
    "optimizer": ["adagrad", "adam"],
    "optimizer_params": {
        "lr": lambda: np.random.uniform(0.0001, 0.01)
    },
    "verbose": True
}

best_model, _, _, _, distMult_mrr_test, _ = select_best_model_ranking(model_class, X_train, X_val, X_test,
                          param_grid,
                          max_combinations=MAX_COMBINATIONS,
                          use_filter=True,
                          verbose=False,
                          early_stopping=False, entities_subset=entities_subset)

save_model(best_model, './trained_models/DistMult.pkl')
test_metrics["DistMult"] = distMult_mrr_test
del best_model


# ## ComplEx
model_class = ComplEx
param_grid = {
    "batches_count": [50],
    "seed": 0,
    "epochs": [10, 50, 100],
    "k": [100, 200],
    "eta": [5, 10, 15],
    "loss": ["pairwise", "nll"],
    "loss_params": {
        "margin": [2]
    },
    "embedding_model_params": {
        
    },
    "regularizer": ["LP", None],
    "regularizer_params": {
        "p": [1, 3],
        "lambda": [1e-4, 1e-5]
    },
    "optimizer": ["adagrad", "adam"],
    "optimizer_params": {
        "lr": lambda: np.random.uniform(0.0001, 0.01)
    },
    "verbose": True
}

best_model, _, _, _, complEx_mrr_test, _ = select_best_model_ranking(model_class, X_train, X_val, X_test,
                          param_grid,
                          max_combinations=MAX_COMBINATIONS,
                          use_filter=True,
                          verbose=False,
                          early_stopping=False, entities_subset=entities_subset)

save_model(best_model, './trained_models/ComplEx.pkl')
test_metrics["ComplEx"] = complEx_mrr_test
#del best_model


# ## Process test metrics

# save test metrics to file
file = open("test_metrics.pkl","wb")
pickle.dump(test_metrics,file)
file.close()

# MRR Score Histogram
names = ["RandomBaseline", "TransE", "DistMult", "ComplEx"]
values = [list(test_metrics["RandomBaseline"].values())[0], 
          list(test_metrics["TransE"].values())[0], 
          list(test_metrics["DistMult"].values())[0], 
          list(test_metrics["ComplEx"].values())[0]]
plt.bar(names, values, color=['red', 'green', 'blue', 'orange'])
plt.ylabel('Score')
plt.xlabel('Model class')
plt.title("MRR score on test data")
plt.savefig("mrr_scores.png");

# MR Score Histogram
names = ["RandomBaseline", "TransE", "DistMult", "ComplEx"]
values = [list(test_metrics["RandomBaseline"].values())[1], 
          list(test_metrics["TransE"].values())[1], 
          list(test_metrics["DistMult"].values())[1], 
          list(test_metrics["ComplEx"].values())[1]]
plt.bar(names, values, color=['red', 'green', 'blue', 'orange'])
plt.ylabel('Score')
plt.xlabel('Model class')
plt.title("MR score on test data");
plt.savefig("mr_scores.png")

# Hits@n Histogram
labels = ['Hits@1','Hits@3','Hits@10']
metrics_RandomBaseline = list(test_metrics["RandomBaseline"].values())[2:5]
metrics_TransE = list(test_metrics["TransE"].values())[2:5]
metrics_DistMult = list(test_metrics["DistMult"].values())[2:5]
metrics_ComplEx = list(test_metrics["ComplEx"].values())[2:5]

x = (np.arange(len(labels)))*3  # the label locations
width = 0.35  # the width of the bars
mpl.style.use("bmh")
fig, ax = plt.subplots()
bar_C11 = ax.bar(x - 3*(width), metrics_RandomBaseline, width, label='RandomBaseline')
bar_C12 = ax.bar(x - 2*(width), metrics_TransE, width, label='TransE')
bar_C21 = ax.bar(x - width, metrics_DistMult, width, label='DistMult')
bar_C22 = ax.bar(x, metrics_ComplEx, width, label='ComplEx')
ax.set_ylabel('Score')
ax.set_title('Hit scores for models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.savefig("hit_scores.png")