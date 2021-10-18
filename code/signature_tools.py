import numpy as np

def get_object_frequencies(dataset):
    objects = dataset[:,0]
    unique, counts = np.unique(objects, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    return frequencies


def get_predicate_frequencies(dataset):
    objects = dataset[:,1]
    unique, counts = np.unique(objects, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    return frequencies


def get_target_frequencies(dataset):
    objects = dataset[:,2]
    unique, counts = np.unique(objects, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    return frequencies


def subset_by_signature(dataset, objects, predicates, targets):
    """
    Extracts a subset of a knowledge graph. It includes all triplets in which the object, predicate and/or target appear in their corresponding lists.
    Not that only one of the three must be present in a triplet for it to be included in the subset.
    
    :param dataset: set of triplets
    :param objects: list of objects
    :param predicastes: list of predicates
    :param targets: list of targets
    """
    #obj_subset =  dataset[np.isin(dataset[:,0], objects)]
    #pred_subset =  dataset[np.isin(dataset[:,1],predicates)]
    #target_subset =  dataset[np.isin(dataset[:,2], targets)]
    #subset_with_duplicates = np.concatenate([obj_subset, pred_subset, target_subset])
    #subset_without_duplicates = np.unique(subset_with_duplicates)
    subset = dataset[np.isin(dataset[:,0], objects) | np.isin(dataset[:,1],predicates) | np.isin(dataset[:,2], targets)]
    return subset


def subset_by_frequency(dataset, min_predicate_freq, min_object_target_freq):
    predicate_freq = get_predicate_frequencies(dataset)
    relevant_predicates = predicate_freq[np.where(predicate_freq[:,1] >= min_predicate_freq)]
    relevant_predicates = relevant_predicates[:,0] # extract list of relevant predicates
    yago_filtered = dataset[np.isin(dataset[:,1], relevant_predicates)]
    object_freq = get_object_frequencies(yago_filtered)
    objects_and_targets = np.concatenate([yago_filtered[:, 0], yago_filtered[:, 2]])
    unique, counts = np.unique(objects_and_targets, return_counts=True)
    object_target_freq = np.asarray((unique, counts)).T
    relevant_obj_target = object_target_freq[np.where(object_target_freq[:,1] >= min_object_target_freq)]
    relevant_obj_target = relevant_obj_target[:,0]
    yago_filtered = yago_filtered[np.isin(yago_filtered[:,0], relevant_obj_target) | np.isin(yago_filtered[:,2], relevant_obj_target)]

    return yago_filtered


def most_frequent_objects(dataset, n = 10):
    """
    Finds the most frequent objects in a dataset of triplets, and returns the n most frequent with their corresponding frequencies.
    
    :param dataset: set of triplets
    :param n: number of most frequent objects to return.
    """
    frequencies = get_object_frequencies(dataset)
    sorted_frequencies = frequencies[frequencies[:, 1].argsort()]
    sorted_frequencies = np.flip(sorted_frequencies)
    
    return sorted_frequencies[:n]
    
    
def most_frequent_predicates(dataset, n = 10):
    """
    Finds the most frequent predicates in a dataset of triplets, and returns the n most frequent with their corresponding frequencies.
    
    :param dataset: set of triplets
    :param n: number of most frequent predicates to return.
    """
    frequencies = get_predicate_frequencies(dataset)
    sorted_frequencies = frequencies[frequencies[:, 1].argsort()]
    sorted_frequencies = np.flip(sorted_frequencies)
    
    return sorted_frequencies[:n]


def most_frequent_targets(dataset, n = 10):
    """
    Finds the most frequent targets in a dataset of triplets, and returns the n most frequent with their corresponding frequencies.
    
    :param dataset: set of triplets
    :param n: number of most frequent targets to return.
    """
    frequencies = get_target_frequencies(dataset)
    sorted_frequencies = frequencies[frequencies[:, 1].argsort()]
    sorted_frequencies = np.flip(sorted_frequencies)
    
    return sorted_frequencies[:n]