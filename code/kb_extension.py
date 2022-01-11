import numpy as np
import pandas as pd
from candidate_generation import generate_candidate_triples
from candidate_ranking import rank_candidate_triples, get_candidates_above_rank
from rule_mining import rule_mining



family_subset = np.loadtxt("family_subset_test.txt", dtype = 'object')
family_relations = ["child", "sibling", "mother", "father", "relative", "spouse"]

def extend_kb(original_kb, model, entity_selection_process, candidate_admittance_criteria):
    # entities = get_entities(entitie_selection_process) TODO
    candidates, entities = generate_candidate_triples(original_kb, entities=None, max_entities=10, relations=family_relations, savefile_name=None)
    print("Candidate generation complete")
    ranked_candidate_triples = rank_candidate_triples(model, candidates, original_kb, entities, savefile_name=None)
    # admitted_candidates = filter_candidates(candidate_admittance_criteria) TODO
    admitted_candidates = get_candidates_above_rank(ranked_candidate_triples, rank_cutoff = 3)
    admitted_candidates = admitted_candidates.drop(columns = ["Sub_rank", "Obj_rank"])
    admitted_candidates_ndarray = admitted_candidates.to_numpy()
    expanded_kb = np.concatenate([original_kb, admitted_candidates_ndarray])
    return expanded_kb

def extend_and_mine(original_kb, model, entity_selection_process, candidate_admittance_criteria):
    """
    Takes a knowledge base and extends it by adding new triples that pass some admittance criteria. Therafter rules are mined from the extended knowledge base.
    
    :param........TODO
    :return rules: Pandas dataframe containing the mined rules and metrics.
    """
    extended_kb = extend_kb(original_kb, model, entity_selection_process, candidate_admittance_criteria)
    rules = rule_mining(kb=original_kb, save_raw_mining_output=False, save_mined_rules=False)
    return rules, extended_kb

"""
# testing
from ampligraph.latent_features import restore_model

model = restore_model('./Wikidata_family_subset_100_epocs.pkl')
candidate_triples = np.loadtxt("delete.txt", dtype = 'object')
entities = np.loadtxt("delete_entities.txt", dtype = 'object')
print(family_subset.shape)
print("\n")

r, ex_kb = extend_and_mine(family_subset, model, entity_selection_process=None, candidate_admittance_criteria=None)
print(r)
print(ex_kb.shape)
"""