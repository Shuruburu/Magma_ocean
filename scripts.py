#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 13:53:11 2025

@author: root
"""

def Classify_the_groups(data):
    keys , columns, values =al.Get_species_elements(data, start_key = "H2_g" , max_elements=14 )
    pressure =al.Integrate(data, keys, "pressure")
    al.Plot_general(data, "O2S_g" , "atmosphere", "volume_mixing_ratio", "pressure", log = True)
    x = np.log10(data["atmosphere"]["pressure"].to_numpy())
    y = np.log10(data["O2S_g"]["volume_mixing_ratio"].to_numpy())
    X , Y = np.meshgrid(x, y)
    X = np.concatenate((np.atleast_2d(x), np.atleast_2d(y)), axis = 0).T
    print(X.shape)
    
    mean, sigma, pi =al.EM_algorithm(X, 4)
    print(mean)
    print(sigma)
    z = al.Give_the_label(X, sigma, mean, pi)
    print(z)
    #al.Plot_the_Key_data(data, "O2S_g", "volume_mixing_ratio" , "pressure" , log= True)
#plan for the things
    #print(data["O2S_g"]["pressure"])
    
    return