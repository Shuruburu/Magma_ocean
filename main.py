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
def Excel_data():
    excel_data, columns = load.Load_the_excel(load.phi100T280())

def Pickle_data():
    return

def Create_a_plot(array):
    return

def Interate():
    return 


# find the expected atmosphere
# plotiing of the shit, some statistocs connected to plotting and the variance 
if __name__=="__main__":
    
    #guess = np.linspace(0, 10, 500)
    #Examples.Monte_Carlo()
    data = load.Load_the_pickle(load.pickle100T280())
    Plots =al.Plots(data)
    keys, columns, values =al.Get_species_elements(data, start_index=0, max_elements=14)
    most =al.classify_the_atmosphere(data, keys, "volume_mixing_ratio")
    filtered =al.Filter_out_data(data, "volume_mixing_ratio", tol = -5)
    Plots.Plot_general_log("CO2_g", "N2_g", "volume_mixing_ratio", "volume_mixing_ratio")
    Plots.Plot_general_log("CO2_g", "H2O_g", "volume_mixing_ratio", "volume_mixing_ratio")
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
    rel_filtered = al.Filter_out_data_relative(data, "volume_mixing_ratio", most, tol = -1)
    
    a =al.Histograms(filtered)
    print("The absolute_treshold {}".format(a))
    a1 =al.Histograms(rel_filtered)
    print("The relative_treshold {}".format(a1))
    print(a)