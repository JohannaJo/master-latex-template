import numpy as np
import pandas as pd
import os
import utils.config as config

def extract_raw_output(filepath):
    """
    Extracts the lines from a text file, deletes the file and returns a list of the lines in the file.
    """
    a_file = open(filepath, "r")
    lines = a_file.readlines()
    a_file.close()
    os.remove(filepath) # remove raw mining output
    return lines


def save_output_to_pd_df(lines):
    # extract header for pandas dataframe
    header_line = lines[0]
    header_line = header_line.rstrip("\n")
    header = header_line.split("\t")
    del lines[0] #remove headers from data
    
    # save data to pandas dataframe
    df_lines = []
    for line in lines:
        df_lines.append(line.rstrip("\n").split("\t"))
    mining_results = pd.DataFrame(df_lines,  columns=header)
    
    # clean up data to have correct format
    mining_results = mining_results.drop(['Functional variable'], axis=1)
    mining_results['PCA Confidence'] = mining_results['PCA Confidence'].apply(lambda x: float(x.replace(',','.')))
    mining_results['Head Coverage'] = mining_results['Head Coverage'].apply(lambda x: float(x.replace(',','.')))
    mining_results['Positive Examples'] = mining_results['Positive Examples'].astype('int')
    mining_results['PCA Body size'] = mining_results['PCA Body size'].astype('int')
    
    return mining_results

def mine_original_rules():
    """
    Mines rules from the original knowledge base
    """ 
    # use AMIE to mine rules    
    mining_output_path = "temp_mined_rules.txt"
    os.system("java -jar " + config.AMIE_miner_filepath + " -bias lazy -full -maxad 3 -noHeuristics -ostd " + config.tsv_kb_path + " > " + mining_output_path)
        
    # save mined rules 
    lines = extract_raw_output(mining_output_path)
    lines = lines[15:-3] # delete the first 15 and last three lines of output file as they do not contain relevant information
    # save mining results
    mining_results = save_output_to_pd_df(lines)
    
    return mining_results


def rule_mining(kb):
    """
     Generates a set of rules mined from a given knowledge base with a given rule mining approach.
     Mines adds metrics calculated on original dataset.
     
    :param kb: nd-array represetning the knowledge base to mine rules from.    
    :param rule_body_size: number of triples permitted in the body of the rule
    :return mining_results: a pandas dataframe containing the rule mining results
    """
    # generate temporary tsv file of kb
    df_kb = pd.DataFrame(kb)
    kb_filepath = "temp_tsv_kb.tsv"
    df_kb.to_csv(kb_filepath, sep="\t", index=None)
        
    # use AMIE to mine rules    
    mining_output_path = "temp_mined_rules.txt"
    os.system("java -jar " + config.AMIE_miner_filepath + " -bias lazy -full -maxad 3 -noHeuristics -ostd " + kb_filepath + " > " + mining_output_path)
    os.remove(kb_filepath) # remove temporary tsv file
        
    # save mined rules 
    lines = extract_raw_output(mining_output_path)
    lines = lines[15:-3] # delete the first 15 and last three lines of output file as they do not contain relevant information

    
    #generate new temp tsv file for metric calculation   
    temp_rule_file_path = "temp_mined_rules.tsv"
    # write mined to tvs file for metric calculation
    new_file = open(temp_rule_file_path, "w+")
    for line in lines:
        new_file.write(line)
    new_file.close()
    
    # save mining results to dataframe
    mining_results = save_output_to_pd_df(lines)
    mining_results = mining_results.drop(["Rule"], axis=1)
    mining_results = mining_results.add_prefix("_") # add prefix to denote metrics are calculated over expanded dataset
    
    # use AMIE to calculate metrics on original kb
    metric_output_path = "temp_metrics.txt"
    os.system("java -cp " + config.AMIE_metrics_filepath + " amie.mining.utils.RecomputeConfidence -ostd :t" + temp_rule_file_path + " " + config.tsv_kb_path + " > " + metric_output_path)
    os.remove(temp_rule_file_path) # remove temporary tsv file
    metric_lines = extract_raw_output(metric_output_path)
    metric_lines = metric_lines[5:-2] # delete the first 5 and last 2 lines of output file as they do not contain relevant information
    metric_results = save_output_to_pd_df(metric_lines)
    
    # combine metrics for original dataset and mining dataset
    combined = pd.concat([metric_results, mining_results], axis = 1)
    
    # get difference in metric calculations
    combined['PCA Diff'] = combined['PCA Confidence'] - combined['_PCA Confidence']
    combined['Pos Diff'] = combined['Positive Examples'] - combined['_Positive Examples']
    combined['Neg Diff'] = (combined['PCA Body size'] - combined['Positive Examples']) - (combined['_PCA Body size'] - combined['_Positive Examples'])
    
    return combined