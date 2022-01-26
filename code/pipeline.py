import numpy as np
import config
import models # load pretrained_models
import pandas as pd
import pickle
import copy
from rule_mining import rule_mining
from kb_extension import extend_kb
from operator import itemgetter
from candidate_generation import generate_candidate_triples
from candidate_ranking import rank_candidates, admit_candidates



# original knowlege base
original_kb = np.loadtxt("family_subset.txt", dtype = 'object')

# mine rules from original knowledge base
original_rules = rule_mining(original_kb)

# convert metrics to correct datatype
original_rules['PCA Confidence'] = original_rules['PCA Confidence'].apply(lambda x: float(x.replace(',','.')))
original_rules['Head Coverage'] = original_rules['Head Coverage'].apply(lambda x: float(x.replace(',','.')))

# save rules mined from original kb
original_rules.to_pickle("./original_rules.pkl")

# parameters
loaded_models = [models.complEx, models.distMult, models.transE, models.randomBaseline]
entity_selection_methods = ["probabilistic", "random", "most_frequent", "least_frequent"]
candidate_admittance_criteria = config.rank_cutoffs #+ config.percents

parameter_combinations= []
for method in entity_selection_methods:
    for model in loaded_models:
        model_name = model.name
        for criteria in candidate_admittance_criteria:
            parameter_combinations.append([method, model_name, criteria])
parameter_combinations = pd.DataFrame(parameter_combinations, columns=["Entity_selection", "Model", "Candidate_criteria"])
total_iterations = len(parameter_combinations)

# save parameter combinations to file
with open("parameter_combinations.pkl", "wb") as file:
    pickle.dump(parameter_combinations, file)

mined_rules = []
kb_extensions = []
iteration_count = 1
extension_sizes = pd.DataFrame([], columns=["Extension", "Entity_selection","Model", "Candidate_criteria"])
for method in entity_selection_methods:
    candidates, entities = generate_candidate_triples(original_kb, config.relations, entities=None, entity_selection_method=method, max_entities=config.max_entities)
    for model in loaded_models:
        ranked_candidate_triples = rank_candidates(model, candidates, original_kb, entities)        
        for criteria in candidate_admittance_criteria:
            admitted_candidates = admit_candidates(ranked_candidate_triples, criteria)
            admitted_candidates = admitted_candidates.drop(columns = ["Sub_rank", "Obj_rank"])
            admitted_candidates = admitted_candidates.to_numpy()
            extended_kb = np.concatenate([original_kb, admitted_candidates])
            rules = rule_mining(extended_kb)
            kb_extensions.append(admitted_candidates)
            # save extension size
            admitted_w_parameters = pd.DataFrame([[len(admitted_candidates), method, model.name, criteria]], columns=["Extension", "Entity_selection", "Model", "Candidate_criteria"])
            extension_sizes = extension_sizes.append(admitted_w_parameters)
            
            mined_rules.append(rules)
            
            # print to keep track of progress
            print(str(iteration_count) + "/" + str(total_iterations))
            iteration_count += 1
            

# convert metrics to correct datatype
for rule_set in mined_rules:
    rule_set['PCA Confidence'] = rule_set['PCA Confidence'].apply(lambda x: float(x.replace(',','.')))
    rule_set['Head Coverage'] = rule_set['Head Coverage'].apply(lambda x: float(x.replace(',','.')))


# save mined rules to file
with open("mined_rules.pkl", "wb") as file:
    pickle.dump(mined_rules, file)


# save kb extensions to file
with open("kb_extensions.pkl", "wb") as file:
    pickle.dump(kb_extensions, file)
    
    
# save size of kb extensions  to file
with open("extension_sizes.pkl", "wb") as file:
    pickle.dump(extension_sizes, file)


# ## Combine to single dataframe
# Combine the list of rule set dataframes to a single large dataframe. Add columns for parameter values used to mine rules.
# generate dataframe that adds information about the parameters used to each row containing a rule
if len(mined_rules) != len(parameter_combinations):
    print("ERROR: number of given parameter combinaitons, " + len(parameter_combinations) + " is not equal to those actually used: " + len(mined_rules))
for i, parameter_row in parameter_combinations.iterrows():
    number_of_rules = len(mined_rules[i])
    parameter_list = parameter_row.values.tolist()
    parameter_full = [copy.deepcopy(parameter_list) for j in range(number_of_rules)]
    parameter_full_df = pd.DataFrame(parameter_full, columns=["Entity_selection", "Model", "Candidate_criteria"])
    mined_rules[i] = pd.concat([mined_rules[i], parameter_full_df], axis=1)

# add original rules to dataframe
number_of_rules = len(original_rules)
parameter_list = ["Original rules","Original rules","Original rules"]
parameter_full = [parameter_list for j in range(number_of_rules)]
parameter_full_df = pd.DataFrame(parameter_full, columns=["Entity_selection", "Model", "Candidate_criteria"])
original_rules_parameters= pd.concat([original_rules, parameter_full_df], axis=1)
mined_rules.append(original_rules_parameters)
mined_rules_parameters = pd.concat(mined_rules)

# combine rule sets into one large dataframe
mined_rules_parameters = pd.concat(mined_rules)

# change datatype to string
mined_rules_parameters['Candidate_criteria'] =  mined_rules_parameters.Candidate_criteria.astype(str)

# save dataframe to file
with open("mined_rules_parameters.pkl", "wb") as file:
    pickle.dump(mined_rules_parameters, file)


print("Rule mining complete")



