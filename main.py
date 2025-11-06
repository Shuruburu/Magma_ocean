# -*- coding: utf-8 -*-

import sys
import numpy as np
import Algorithms as al
import matplotlib.pyplot as plt
#from petitRADTRANS.radtrans import Radtrans
#from petitRADTRANS import physical_constants as cst
import load
import pandas as pd
import scripts
import sys
np.set_printoptions(threshold=sys.maxsize)


# find the expected atmosphere
# plotiing of the shit, some statistocs connected to plotting and the variance 
def Selection():
    data = load.Load_the_pickle(load.pickle100T280())
    Plots =al.Plots(data)
    keys, columns, values =al.Get_species_elements(data, start_index=0, max_elements=14)
    most = al.classify_the_atmosphere(data, keys, "volume_mixing_ratio")
    
    filtered_data = al.Filter_out_data(data, "volume_mixing_ratio", tol =-16 )
    rel_filtered = al.Filter_out_data_relative(data, "volume_mixing_ratio", most)
    
    
    selected = al.Seleciton(rel_filtered)
    selected1 = al.Seleciton(filtered_data)
    
    preselected , preselected1=al.Select_unique(selected, selected1)


    return preselected, most, preselected1

def Simulate(keys ,arg):
    # zrobic dwa scrypty, w ktorym jednym dostaje klucze do tablicy a w drugim iteruje po tych kluczach
    # taki ma byc plan na ta symulacje
    return
def Bash_keys(keys):
    for key in keys:
        print(key, end= " ")
    
if __name__=="__main__":
    
    #guess = np.linspace(0, 10, 500)
    #Examples.Monte_Carlo()
    data = load.Load_the_pickle(load.pickle100T280())
    Plots =al.Plots(data)
    ratio =al.Element_ratios(data)
    
    preselected, most, preselected1 = Selection()
    print(preselected)
    print("==================================")
    print(preselected1)
    selected = al.Selection_of_mixing(preselected1)
    print(selected)
    mixing_ratio =  al.Mixing_ratio(data, "volume_mixing_ratio" , selected)
    to_be_writen_file =al.Add_the_domiant_species(mixing_ratio, most)

    #Here the plan is to pass the dictionary with the number of the atmosphere that we acquired
    lines = load.Write_In_the_file("/home/shurubura/Documents/VULCAN/vulcan_cfg.py", to_be_writen_file[0])
    
    if sys.argv == "get_the_keys":
        Bash_keys(mixing_ratio.keys())
    