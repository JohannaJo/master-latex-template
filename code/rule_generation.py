import numpy as np
import os
import pandas as pd

family_subset = np.loadtxt("family_subset_test.txt", dtype = 'object')

def rule_set_generation(kb_filepath = "family_subset_tsv_test.tsv", miner_filepath = "amie-milestone-intKB.jar", max_rule_body_size = 2, save_raw_mining_output = False, save_mined_rules = False):
    """
     Generates a set of rules mined from a given knowledge base with a given rule mining approach.
    
    :param kb_filepath: nd-array represetning the knowledge base to mine rules from.
    :param miner_filepath: rule mining approach being used
    :param rule_body_size: number of triples permitted in the body of the rule # TODO: implement this parameter
    :return mining_results: a pandas dataframe containing the rule mining results
    """
    mining_output_path = "family_mined_rules_test.txt"
    max_rule_size = max_rule_body_size + 1
    os.system("java -jar " + miner_filepath + " -bias lazy -full -maxad " + str(max_rule_size) + " -noHeuristics -ostd " + kb_filepath + " > " + mining_output_path)
    
    a_file = open(mining_output_path, "r")
    lines = a_file.readlines()
    a_file.close()
    if not(save_raw_mining_output):
        os.remove(mining_output_path)
    
    # delete the first 15 and last three lines as they do not contain relevant information
    for index in range(15):
        del lines[0]
    for index in range(3):
        del lines[-1]    
        
    # extract header for pandas dataframe
    header_line = lines[0]
    header_line = header_line.rstrip("\n")
    header = header_line.split("\t")
    
    # write header to file
    if save_mined_rules:
        new_file = open("rules_test.txt", "w+")
        new_file.write(lines[0])
    
    del lines[0] #remove headers from data

    # save data to pandas dataframe and text file
    df_lines = []
    for line in lines:
        df_lines.append(line.rstrip("\n").split("\t"))
    if save_mined_rules:
        for line in lines:
            new_file.write(line)
        new_file.close()
    
    mining_results = pd.DataFrame(df_lines,  columns = header)
    return mining_results

print(rule_set_generation(save_raw_mining_output = False, save_mined_rules = True))