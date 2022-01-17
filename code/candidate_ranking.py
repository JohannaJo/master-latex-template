import numpy as np
import pandas as pd
from ampligraph.evaluation import evaluate_performance
from ampligraph.latent_features import restore_model

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
    
    
def get_candidates_above_rank(ranked_candidates, rank_cutoff: int=3):
    """
    Filters out all candidates that have either a subject rank or an object rank below the rank cutoff.
    
    :param ranked_candidates: pandas dataframe containing ranked triples.
    :param rank_cutoff: the lower limit for acceptable rank when filtering candidates.
    :return top_ranked_candidates: a pandas dataframe containing only the candidates with subject and object rank above the cutoff value.
    """
    top_ranked_candidates = ranked_candidates.loc[(ranked_candidates["Sub_rank"] <= rank_cutoff) & (ranked_candidates["Obj_rank"] <= rank_cutoff)]
    return top_ranked_candidates


def get_top_n_percent(ranked_candidates, n=10):
    number_of_candidates = len(ranked_candidates)
    index_cutoff = int(number_of_candidates * (n/100))
    ranked_candidates = ranked_candidates.sort_values(["Sub_rank", "Obj_rank"])
    top_n_percent =  ranked_candidates.head(index_cutoff)
    return top_n_percent


def admit_candidates(ranked_candidates, admittance_criteria):
    admittance_type = admittance_criteria[0]
    if admittance_type == "percent":
        return get_top_n_percent(ranked_candidates, admittance_criteria[1])
    elif admittance_type == "rank_cutoff":
        return get_candidates_above_rank(ranked_candidates, admittance_criteria[1])
    else:
        print("Admittance criteria \"", str(admittance_criteria), "\" is not valid.")
        return None
    


"""
TESTING  

family_subset = np.loadtxt("family_subset_test.txt", dtype = 'object')

model = restore_model('./Wikidata_family_subset_100_epocs.pkl')
candidate_triples = np.loadtxt("delete.txt", dtype = 'object')
entities = np.loadtxt("delete_entities.txt", dtype = 'object')

ranked = rank_candidate_triples(model, candidate_triples, family_subset, entities, "delete_ranks")
print(get_candidates_above_rank(ranked, 5))
"""