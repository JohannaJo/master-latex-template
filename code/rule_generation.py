def rule_set_generation(kb, mining_method, rule_body_size):
     """
     Generates a set of rules mined from a given knowledge base with a given rule mining approach.
    
    :param kb: nd-array represetning the knowledge base to mine rules from.
    :param mining_method: rule mining approach being used
    :param rule_body_size: number of triples permitted in the body of the rule
    :return mining_results: a pandas dataframe containing the rule mining results
    """
        