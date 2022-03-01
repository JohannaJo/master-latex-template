"""
Contains global variables of loaded trained models. Ensure that models to be loaded in fact are present in the same directory as this file.
"""
from ampligraph.latent_features import restore_model
import os.path as path

current_dir = path.dirname(path.realpath(__file__))
parent_dir =  path.abspath(path.join(current_dir ,".."))
trained_models_dir = path.abspath(path.join(parent_dir , "models/trained_models"))

def set_models_to_wn18rr():
    trained_models_path = path.abspath(path.join(trained_models_dir , "wn18rr_25_combinations"))
    global randomBaseline
    global transE
    global distMult
    global complEx
    randomBaseline = restore_model(trained_models_path + '/RandomBaseline.pkl')
    transE = restore_model(trained_models_path + '/TransE.pkl')
    distMult = restore_model(trained_models_path + '/DistMult.pkl')
    complEx = restore_model(trained_models_path + '/ComplEx.pkl')
    
def set_models_to_family_dataset():
    trained_models_path = path.abspath(path.join(trained_models_dir , "family_25_combinations"))
    global randomBaseline
    global transE
    global distMult
    global complEx
    randomBaseline = restore_model(trained_models_path + '/RandomBaseline.pkl')
    transE = restore_model(trained_models_path +'/TransE.pkl')
    distMult = restore_model(trained_models_path + '/DistMult.pkl')
    complEx = restore_model(trained_models_path + '/ComplEx.pkl')