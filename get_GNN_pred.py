"""
##### main file for GNN lignin solubility prediction #####

1. enter smiles of solvent in pred_mols.csv
2. run code
3. open pred_mols.csv to see results

author: laura koenig-mattern
mail: koenig-mattern@mpi-magdeburg.mpg.de

GNN code by: Edgar Ivan Sanchez Medina
mail: sanchez@mpi-magdeburg.mpg.de

"""


import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit import RDConfig
import networkx as nx
import copy
import re
import thermo
import Trained_GNN_LC.GNN_LC as GNN
import matplotlib.pyplot as plt
import time
from rdkit.Chem import Descriptors


path_smi_frame = 'pred_mols.csv'
df_pred = pd.read_csv(path_smi_frame, encoding='cp1252', sep=',', index_col=0)
df_pred['mol'] = df_pred['SMILES'].apply(Chem.MolFromSmiles)
df_pred['mol wt'] = df_pred['mol'].apply(Chem.Descriptors.MolWt)
df_pred = GNN.GNN_lignin(df_pred, 'mol')
mol_wt_GGG = 530.57 # g/mol
df_pred['sol wt percent'] = (mol_wt_GGG*(10**df_pred['GNN_prediction_lignin']) / (mol_wt_GGG*(10**df_pred['GNN_prediction_lignin']) + (1-(10**df_pred['GNN_prediction_lignin']))*df_pred['mol wt']))*100

df_pred.to_csv(path_smi_frame)

#df_pred.to_csv('C:/Users/laura/PycharmProjects/Python/GA/results/1000_mols_1000_gens_fewer_children_new_frags_max_molwt_new_operations_2/figures_highlighted_ES/LKM/230227_sol_des_pred_all.csv')
