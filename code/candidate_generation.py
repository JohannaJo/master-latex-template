import numpy as np
import pandas as pd
from signature_tools import *

# used to set default parameter
family_relations = ["child", "sibling", "mother", "father", "relative", "spouse"]


def get_entities(kb, max_entities: int = 1000, selection_method: str="random", savefile_name=None):
    """
    Generates a numpy array of entities selected by a given method.
    
    :param kb: numpy nd-array representing the knowledge base to extract the most common entities from.
    :param max_entities: upper limit to number of entities to return.
    
    :return entities_subset: numpy array of a subset of entities in the knowledge base.
    """
    obj_target_max = max_entities//2
    if selection_method == "random":
        object_entities = random_objects(kb, n=obj_target_max)
        subject_entities = random_targets(kb, n=obj_target_max)
    else:
        if selection_method == "most_frequent":
            selected_objects = most_frequent_objects(kb, n=obj_target_max)
            selected_targets = most_frequent_targets(kb, n=obj_target_max)
        elif selection_method == "least_frequent":
            selected_objects = least_frequent_objects(kb, n=obj_target_max)
            selected_targets = least_frequent_targets(kb, n=obj_target_max)
        else:
            # check for invalid input
            print("Selection method \"" + selection_method + "\" is not valid.")
            return None
        object_entities = selected_objects[:,0]
        subject_entities = selected_targets[:,0]
    
    # generate array of unique selected entities
    entities_subset = np.concatenate([subject_entities, object_entities[~np.isin(object_entities,subject_entities)]])
    
    if savefile_name is not None:
        pd.DataFrame(entities_subset).to_csv(savefile_name, sep = "\t", header=None, index=None)
        
    return entities_subset


def generate_candidate_triples(kb, entities=None, entity_selection_method:str="random", max_entities=100, relations=family_relations, savefile_name = None):
    if entities is None:
        # generate a list of the most common entities
        entities_subset = get_entities(kb, max_entities, entity_selection_method)
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
    
    if savefile_name is not None:
        pd.DataFrame(candidate_triples).to_csv(savefile_name + ".txt", sep = "\t", header=None, index=None)
        
    return candidate_triples, entities_subset
 
# testing 
#family_subset = np.loadtxt("family_subset_test.txt", dtype = 'object')
#print("Why is this line running?????")    
#print(get_most_common_entities(family_subset, max_entities = 10, savefile_name = "delete_entities"))