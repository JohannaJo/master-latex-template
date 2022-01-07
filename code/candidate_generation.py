import numpy as np
import pandas as pd
from signature_tools import most_frequent_objects, most_frequent_targets

family_subset = np.loadtxt("family_subset_test.txt", dtype = 'object')

def get_most_common_entities(kb, max_entities: int = 1000):
    """
    Generates a numpy array of the most common entities in a set of triples.
    
    :param kb: numpy nd-array representing the knowledge base to extract the most common entities from.
    :param max_entities: upper limit to number of entities to return.
    
    :return entities_subset: numpy array of the most common entities in the knowledge base.
    """
    obj_target_max = max_entities//2
    top_objects = most_frequent_objects(kb, n=obj_target_max)
    top_targets = most_frequent_targets(kb, n=obj_target_max)
    
    subject_entities = top_targets[:,1]
    object_entities = top_objects[:,1]
    
    entities_subset = np.concatenate([subject_entities, object_entities[~np.isin(object_entities,subject_entities)]])
    return entities_subset

def generate_candidate_triples(kb, entities=None, max_entities=100, relations=["child", "sibling", "mother", "father", "relative", "spouse"], candidates_file_name = None):
    
    if entities is None:
        # generate a list of the most common entities
        entities_subset = get_most_common_entities(kb, max_entities)
    else:
        entities_subset = entities

    # generate all possible triple combinations with relations and top entities
    candidate_triples_unfiltered = np.array(np.meshgrid(entities_subset, relations, entities_subset)).T.reshape(-1,3)
    
    # convert list of triples to hashable set of strings, this is so that they can be compared to the family subset
    candidate_triples_unfiltered_set = set([' '.join(map(str, triple)) for triple in candidate_triples_unfiltered])

    kb_set = set([' '.join(map(str, triple)) for triple in kb])
    
    # remove candidate triples that already exist in the dataset
    candidate_triples_set = candidate_triples_unfiltered_set - kb_set

    # convert candidate triple set back to numpy ndarray format
    candidate_triples = np.array([list(triple_string.split(" ")) for triple_string in candidate_triples_set])
    
    if candidates_file_name is not None:
        pd.DataFrame(candidate_triples).to_csv(candidates_file_name + ".txt", sep = "\t", header=None, index=None)
        
    return candidate_triples
    
    
print(generate_candidate_triples(family_subset, max_entities = 10, candidates_file_name = "delete"))