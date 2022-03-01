"""
Contains global variables that are used across modules.
"""
from utils.models import set_models_to_wn18rr, set_models_to_family_dataset
import numpy as np

# RULE MINING PARAMETERS

AMIE_miner_filepath = "amie-milestone-intKB.jar"
AMIE_metrics_filepath = "amie-custom-KB.jar"

# generate candidate admittance criteria for hyperparameter search
rank_cutoffs = [("rank_cutoff", i) for i in range(1, 10, 3)]

#candidate_generation_strategies = ['random_uniform', 'entity_frequency', 'graph_degree', 'cluster_coefficient', 'cluster_triangles', 'cluster_squares']
candidate_generation_strategies = ["probabilistic", "random", "most_frequent", "least_frequent"]
max_entities = 1000
#max_candidates_per_relation = 50000

# family subset
family_relations = ["child", "sibling", "mother", "father", "relative", "spouse"]
family_dataset_path = "datasets/family_subset.txt"
family_kb_tsv_path = "datasets/family_kb.tsv"

# WIN18RR subset
wn18rr_dataset_path = "datasets/wn18rr_subset.txt"
wn18rr_relations = ['_hypernym', '_derivationally_related_form', '_member_meronym', '_has_part', '_synset_domain_topic_of', '_instance_hypernym']
wn18rr_kb_tsv_path = "datasets/wn18rr_kb.tsv"



#COMMENT AND UNCOMMENT FOR THE RULE MINING RUNS YOU WANT

#dataset_path = wn18rr_dataset_path
#relations = wn18rr_relations
#tsv_kb_path = wn18rr_kb_tsv_path
#set_models_to_wn18rr()

relations = family_relations
dataset_path = family_dataset_path
tsv_kb_path = family_kb_tsv_path
set_models_to_family_dataset()




# MODEL SELECTION PARAMETERS
val_test_size = 0.2
test_size = 0.1
max_cobinations = 25
max_training_entities = 1000
#model_selection_data_path = wn18rr_dataset_path
model_selection_data_path = family_dataset_path
param_grid = {
    "batches_count": [50, 100],
    "seed": 0,
    "epochs": [50, 100],
    "k": [50, 100, 200],
    "eta": [5, 10, 15],
    "loss": ["nll","pairwise"],
    "loss_params": {
        "margin": [0.5, 1, 2]
    },
    "embedding_model_params": {
        
    },
    "regularizer": ["LP"],
    "regularizer_params": {
    },
    "optimizer": ["adam"],
    "optimizer_params": {
        "lr": lambda: np.random.uniform(0.0001, 0.01)
    },
    "verbose": True
}