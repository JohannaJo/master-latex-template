"""
Contains global variables of loaded trained models. Ensure that models to be loaded in fact are present in the same directory as this file.
"""
from ampligraph.latent_features import restore_model


# load pretrained knowledge graph embeddings
randomBaseline = restore_model('./RandomBaseline.pkl')
transE = restore_model('./TransE.pkl')
distMult = restore_model('./DistMult.pkl')
complEx = restore_model('./ComplEx.pkl')