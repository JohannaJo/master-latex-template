import csv
import pandas as pd

def parse_rule(string_rule: str):
    """
    Parses rule in the format given by AMIE+.
    Example input: '?b  _has_part  ?a   => ?a  _synset_domain_topic_of  ?b'
    Example output: 
        body_parts:[['?b', '_has_part', '?a']]
        head: ['?a', '_synset_domain_topic_of', '?b']
        
    """
    list_rule = string_rule.split()
    body_1 = [list_rule[0], list_rule[1], list_rule[2]]
    body_parts = [body_1]
    if len(list_rule) > 7:
        body_2 = [list_rule[3], list_rule[4], list_rule[5]]
        body_parts.append(body_2)
        head = [list_rule[7], list_rule[8], list_rule[9]]
    else:
        head = [list_rule[4], list_rule[5], list_rule[6]]
    return body_parts, head


def generate_rule_tsv_file(mined_rules_df, filename='test'):
    """
    Generates tsv file of mined rules.
    
    :param mined_rules_df: pandas dataframe of output rules and metrics.
    """
    rules = [x.split() for x in mined_rules_df.Rule.values]

    with open(filename + ".tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter = '\t')
        writer.writerows(rules)
        
        
def generate_kb_tsv_file(kb, filename='test'):
    df_kb = pd.DataFrame(kb)
    df_kb.to_csv(filename + ".tsv", sep="\t", index=None)