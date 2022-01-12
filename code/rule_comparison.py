import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def combine_rules(rule_set_A, rule_set_B):
    return pd.merge(rule_set_A, rule_set_B, how='outer', on=['Rule'])


def get_common_rules(rule_set_A, rule_set_B):
    return pd.merge(rule_set_A, rule_set_B, how='inner', on=['Rule', "Functional_variable"])
    
    
def get_unique_rules(rule_set_A, rule_set_B):
    unique_A = rule_set_A[~rule_set_A.Rule.isin(rule_set_B.Rule)]
    unique_B = rule_set_B[~rule_set_B.Rule.isin(rule_set_A.Rule)]
    return unique_A, unique_B


def plot_pie_chart(rule_set_A, rule_set_B):
    common = get_common_rules(rule_set_A, rule_set_B)
    unique_A, unique_B = get_unique_rules(rule_set_A, rule_set_B)
    
    # plot distribution of the mined rules
    data = [len(common), len(unique_A), len(unique_B)]
    labels = ['Common for both mined sets', 'Unique to original', 'Unique to expanded']

    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:5]

    #create pie chart
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    plt.title("Distribution of rules mined")
    plt.show()

