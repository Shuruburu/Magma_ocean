#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 13:53:11 2025

@author: root
"""

import sys
import numpy as np
import Algorithms as al
import matplotlib.pyplot as plt
#from petitRADTRANS.radtrans import Radtrans
#from petitRADTRANS import physical_constants as cst
import load
import pandas as pd
import sys


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
def Filter_approach(data):
    Plots=al.Plots(data)
    keys, columns, values =al.Get_species_elements(data, start_index=0, max_elements=14)
    most =al.classify_the_atmosphere(data, keys, "volume_mixing_ratio")
    filtered =al.Filter_out_data(data, "volume_mixing_ratio", tol = -5)
    Plots.Plot_general_log("CO2_g", "N2_g", "volume_mixing_ratio", "volume_mixing_ratio")
    Plots.Plot_general_log("CO2_g", "ClH_g", "volume_mixing_ratio", "volume_mixing_ratio")
    Plots.Plot_filtered(
        "CO2_g",
        "N2_g",
        "volume_mixing_ratio",
        [filtered["CO2_g"], filtered["N2_g"]] , filtered["H2O_g"]
        )
    Plots.Plot_filtered(
        "CO2_g",
        "H2O_g",
        "volume_mixing_ratio",
        [filtered["CO2_g"], filtered["H2O_g"]], filtered["N2_g"]
    )

    Plots.Plot_filtered(
        "N2_g",
        "H2O_g",
        "volume_mixing_ratio",
        [filtered["N2_g"], filtered["H2O_g"]] ,filtered["CO2_g"]
    )
    rel_filtered = al.Filter_out_data_relative(data, "volume_mixing_ratio", most)
    
    bins1 =al.Histograms(filtered)
    Plots.Histograms(bins1)
    print("The absolute_treshold {}".format(bins1))
    bins =al.Histograms(rel_filtered)
    Plots.Histograms(bins, name = "relative")
    print("The relative_treshold {}".format(bins))
    print(bins)