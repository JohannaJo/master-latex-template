"""
Contains global variables that are used across modules.
"""
# generate candidate admittance criteria for hyperparameter search
rank_cutoffs = [("rank_cutoff", i) for i in range(1, 10, 3)]
percents = [("percent", i) for i in range(1, 11,5)]
max_entities = 500