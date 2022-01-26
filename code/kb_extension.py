import numpy as np
import pandas as pd
from candidate_generation import generate_candidate_triples
from candidate_ranking import rank_candidates, get_candidates_above_rank, admit_candidates
from rule_mining import rule_mining

# used as default parameter
family_relations = ["child", "sibling", "mother", "father", "relative", "spouse"]

def extend_kb(original_kb, model, entity_selection_method, candidate_admittance_criteria, max_entities: int):
    """
    Takes a knowledge base and extends it by adding new triples that pass some admittance criteria.
    
    :param original_kb: numpy ndarray representing the original knowledge base.
    :param model: knowledge graph embedding trained on the original knowledge base.
    :param entity_selection_method: parameter for how entities are generated.
    :param candidate_admittance_criteria: parameter for filtering out triples not deemed good enough for addition to the kb.
    :param max_entities: maximum number of entities to generate candidates from.
    :return expanded_kb: numpy ndarray containing the original kb with additional new triples, ie an extended version of the original kb.
    :return admitted_candidates: numpy ndarray containing the additional new triples that were added to the extended kb.
    """
    candidates, entities = generate_candidate_triples(original_kb, entities=None, entity_selection_method=entity_selection_method, max_entities=max_entities, relations=family_relations, savefile_name=None)
    ranked_candidate_triples = rank_candidates(model, candidates, original_kb, entities, savefile_name=None)
    admitted_candidates = admit_candidates(ranked_candidate_triples, candidate_admittance_criteria)
    admitted_candidates = admitted_candidates.drop(columns = ["Sub_rank", "Obj_rank"])
    admitted_candidates = admitted_candidates.to_numpy()
    expanded_kb = np.concatenate([original_kb, admitted_candidates])
    return expanded_kb, admitted_candidates

def extend_and_mine(original_kb, model, entity_selection_method, candidate_admittance_criteria):
    """
    Takes a knowledge base and extends it by adding new triples that pass some admittance criteria. Therafter rules are mined from the extended knowledge base.
    
    :param original_kb: numpy ndarray representing the original knowledge base.
    :param model: knowledge graph embedding trained on the original knowledge base.
    :param entity_selection_method: parameter for how entities are generated.
    :param candidate_admittance_criteria: parameter for filtering out triples not deemed good enough for addition to the kb.
    :return rules: Pandas dataframe containing the mined rules and metrics.
    """
    extended_kb, admitted_candidates = extend_kb(original_kb, model, entity_selection_method, candidate_admittance_criteria)
    rules = rule_mining(kb=original_kb, save_raw_mining_output=False, save_mined_rules=False)
    return rules, extended_kb

"""
# testing
from ampligraph.latent_features import restore_model

family_subset = np.loadtxt("family_subset_test.txt", dtype = 'object')
model = restore_model('./Wikidata_family_subset_100_epocs.pkl')
candidate_triples = np.loadtxt("delete.txt", dtype = 'object')
entities = np.loadtxt("delete_entities.txt", dtype = 'object')
print(family_subset.shape)
print("\n")

r, ex_kb = extend_and_mine(family_subset, model, entity_selection_method=None, candidate_admittance_criteria=None)
print(r)
print(ex_kb.shape)
"""