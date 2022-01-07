import numpy as np
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

def generate_candidate_triples(kb, max_entities=100, relations=["child", "sibling", "parent", "relative", "spouse"]):
    # generate a list of the most common entities
    top_objects = most_frequent_objects(family_subset, n=2500)
    top_targets = most_frequent_targets(family_subset, n=2500)
    subject_entities = top_targets[:,1]
    object_entities = top_objects[:,1]
    entities_subset = np.concatenate([subject_entities, object_entities[~np.isin(object_entities,subject_entities)]])
    
    
print(get_most_common_entities(family_subset, max_entities = 100))