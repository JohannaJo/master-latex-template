import numpy as np
import os
import pandas as pd

def rule_set_generation(kb=None, kb_filepath=None, miner_filepath="amie-milestone-intKB.jar", max_rule_body_size=2, save_raw_mining_output=False, save_mined_rules=False):
    """
     Generates a set of rules mined from a given knowledge base with a given rule mining approach.
     
    :param kb: nd-array represetning the knowledge base to mine rules from.    
    :param kb_filepath: name of a tsv file that 
    :param miner_filepath: rule mining approach being used
    :param rule_body_size: number of triples permitted in the body of the rule
    :return mining_results: a pandas dataframe containing the rule mining results
    """
    # check for invalid input
    if (kb is None) and (kb_filepath is None):
        print("Error: no kb object or filepath is given.")
        return None
    
    # generate tsv file if none is given
    delete_generated_tsv = False
    if kb_filepath is None:
        df_kb = pd.DataFrame(kb)
        kb_filepath = "temp_tsv_kb.tsv"
        df_kb.to_csv(kb_filepath, sep="\t", index=None)
        delete_generated_tsv = True
        
    mining_output_path = "family_mined_rules_test.txt"
    max_rule_size = max_rule_body_size + 1
    os.system("java -jar " + miner_filepath + " -bias lazy -full -maxad " + str(max_rule_size) + " -noHeuristics -ostd " + kb_filepath + " > " + mining_output_path)
    
    if delete_generated_tsv:
        os.remove(kb_filepath)
        
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


# testing
family_subset = np.loadtxt("family_subset_test.txt", dtype = 'object')
print(rule_set_generation(kb=family_subset, save_raw_mining_output=False, save_mined_rules=True))