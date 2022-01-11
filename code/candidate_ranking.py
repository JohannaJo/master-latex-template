import numpy as np
import pandas as pd
from ampligraph.evaluation import evaluate_performance
from ampligraph.latent_features import restore_model

family_subset = np.loadtxt("family_subset_test.txt", dtype = 'object')

def rank_candidate_triples(model, candidates, kb, entities, savefile_name: str=None):
    # find ranks for subset
    ranks = evaluate_performance(candidates, model = model, filter_triples=kb, entities_subset = entities)
    
    # save ranked candidates in Pandas DataFrame
    data = {"Object" : candidates[:,0],
            "Predicate" : candidates[:,1],
            "Subject" : candidates[:,2],
            "Sub_rank": ranks[:,0],
            "Obj_rank": ranks[:,1]
           }
    ranked_candidates = pd.DataFrame(data)
    # save dataframe
    if savefile_name is not None:
        ranked_candidates.to_pickle(savefile_name + ".pkl")
        
    return ranked_candidates
    
    
def get_candidates_above_rank(ranked_candidates, rank_cutoff: int):
    """
    Filters out all candidates that have either a subject rank or an object rank below the rank cutoff.
    
    :param ranked_candidates: pandas dataframe containing ranked triples.
    :param rank_cutoff: the lower limit for acceptable rank when filtering candidates.
    :return top_ranked_candidates: a pandas dataframe containing only the candidates with subject and object rank above the cutoff value.
    """
    top_ranked_candidates = ranked_candidates.loc[(ranked_candidates["Sub_rank"] <= rank_cutoff) & (ranked_candidates["Obj_rank"] <= rank_cutoff)]
    return top_ranked_candidates



"""
TESTING  
model = restore_model('./Wikidata_family_subset_100_epocs.pkl')
candidate_triples = np.loadtxt("delete.txt", dtype = 'object')
entities = np.loadtxt("delete_entities.txt", dtype = 'object')

ranked = rank_candidate_triples(model, candidate_triples, family_subset, entities, "delete_ranks")
print(get_candidates_above_rank(ranked, 5))
"""