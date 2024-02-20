'''
PSEvolve main file: lignin solvent design

author: laura koenig-mattern
mail: koenig-mattern@mpi-magdeburg.mpg.de

date: Feb 19, 2024
'''

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit import RDConfig
import Trained_GNN_LC.GNN_LC as GNN
import utils as u

# set hyperparams
pop_size = 1000
generations = 100
mutation_rate = 0.1
num_parents = 50
num_children = num_parents*2
max_mol_weight = 200 # g/mol
group_constraints = 'AAF' #None, 'AAF'
SA_max = 3.5

# save dataframes of all postprocessed populations: 0 - false, 1- true
save_all_pops_flag = 0

# initialize start population
df_pop = u.generate_start_pop_hexane(pop_size)

## initalize frame to store the fittest molecules
df_fittest = df_pop

# initialize statistics frame
df_stats = u.ini_statistics_frame()

# initialize group frame
df_groups = u.ini_group_frame()

# # mols generated
# smi_list = []
for gen in range(generations):
    print('------ generation %s is running ------' %(str(gen + 1)))

    # select the fittest and perform cross over
    print('start cross-over')
    df_pop = u.cross_over(df_pop, num_parents, num_children, max_mol_weight, group_constraints, SA_max)

    # perform mutation
    print('start mutation')
    df_pop = u.mutation(df_pop, mutation_rate, max_mol_weight, group_constraints, SA_max)

    # get solubilities
    print('get solubilties')
    df_pop = GNN.GNN_lignin(df_pop, 'mol')

    print('get SA scores')
    df_pop = u.get_SA_scores(df_pop)

    # get fitness vals
    print('get fitness vals')
    df_pop = u.get_fitness_vals(df_pop)

    # # save generated mols in smi list
    # smi_pop = df_pop['smi'].to_list()
    # for smi in smi_pop:
    #     smi_list.append(smi)
    # smi_list = list(set(smi_list))


    # sort the table according to descending fitness and only take the pop_size fittest
    print('sort pop and delete weakest')
    df_pop = df_pop.sort_values(by=['fitness value'], ascending=False)
    df_pop = df_pop.head(pop_size)


    ###### for postprocessing #####

    # get fittest molecules of population and save
    print('save fittest')
    df_fittest = u.get_all_time_fittest(df_fittest, df_pop)

    # get stats
    print('get stats')
    df_stats = u.get_stats(df_stats, df_pop)

    # get group counts
    print('get group counts')
    df_groups = u.get_group_counts(df_groups, df_pop)


    # post-process every 5 generations
    if gen % 5 == 0:
        print('post process')
        u.post_process(df_pop, df_fittest, df_stats, df_groups, save_all_pops_flag, gen)

# save_path = './results/frames/'
# df_pop.to_csv(save_path + 'last_pop.csv')
# df_fittest.to_csv(save_path + 'fittest.csv')
# df_stats.to_csv(save_path + 'stats.csv')
# df_groups.to_csv(save_path + 'group_counts.csv')







