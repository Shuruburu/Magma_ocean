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

#NO H2_g
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



def Create_the_grid(number):
    # first, make pressure grid
    K_z  =  100
    p_surf    = data['atmosphere']['pressure'][number]*1e6
    logp_surf = np.log10(p_surf)
    logp      = np.linspace(logp_surf,-8, 100)
    pressures = 10**logp
    T_eq = data['atmosphere']['temperature'][0]
    tempretures=  T_eq*np.ones(pressures.size)
    eddy = np.ones(pressures.size)*K_z
    z = np.arange(0, pressures.size)
    return pressures , tempretures, z ,eddy



def Simulate(keys ,arg):
    # zrobic dwa scrypty, w ktorym jednym dostaje klucze do tablicy a w drugim iteruje po tych kluczach
    # taki ma byc plan na ta symulacje
    return
def Bash_keys(keys):
    for key in keys:
        print(key, end= " ")
    
if __name__== "__main1__":
    
    #guess = np.linspace(0, 10, 500)
    #Examples.Monte_Carlo()
    data = load.Load_the_pickle(load.pickle100T280())
    Plots =al.Plots(data)
    ratio =al.Element_ratios(data)
    preselected, most, preselected1 = Selection()
    selected = al.Selection_of_mixing(preselected1)
    mixing_ratio =  al.Mixing_ratio(data, "volume_mixing_ratio" , selected)
    print(mixing_ratio)
    #Here the plan is to pass the dictionary with the number of the atmosphere that we acquired
    #What bases that the vulcan have, adjust to make the next abundant to fill the gap? 
    #Impreove the script to make sure that the mixing is getting renamed after so I would be easier to have more plots
    #Check for the errors of the second simulation
    if len(sys.argv) > 1:
        if sys.argv[1] =="Species":
            to_be_writen_file =al.Add_the_domiant_species(mixing_ratio, most)
              
            #Missing the P_b file_edition while due to change of the pressure file 
            pressure , tempretures,z, eddy =Create_the_grid(int(sys.argv[2]))
            load.Eddit_Kzz(tempretures, pressure, eddy, z)
            lines = load.Write_In_the_file("/home/shurubura/Documents/VULCAN/vulcan_cfg.py", to_be_writen_file[int(sys.argv[2])], pressure[0], int(sys.argv[2]) ) 
            
        elif sys.argv[1] == "get_the_keys":
            Bash_keys(mixing_ratio.keys())

    