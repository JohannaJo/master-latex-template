import numpy as np
import pandas as pd
from signature_tools import most_frequent_objects, most_frequent_targets

family_subset = np.loadtxt("family_subset_test.txt", dtype = 'object')

def rank_candidate_triples(model, candidates, kb, entities, savefile_name = None):
    # find ranks for subset
    ranks = evaluate_performance(candidates, model = model, filter_triples = kb, entities_subset = entities)
    
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
        ranked_candidates.to_pickle(savefile_name)
        
    return ranked_candidates
    
    
    
    
    
    
model = restore_model('./Wikidata_family_subset_100_epocs.pkl')
candidate_triples = np.loadtxt("delete.txt", dtype = 'object')

print(rank_candidate_triples(model, candidate_triples,))