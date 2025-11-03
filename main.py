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
    keys, columns, values =al.Get_species_elements(data, start_index=0, max_elements=14)
    most =al.classify_the_atmosphere(data, keys, "volume_mixing_ratio")
    print(most)
    filtered =al.Filter_out_data(data, "volume_mixing_ratio")

    rel_filtered = al.Filter_out_data_relative(data, "volume_mixing_ratio", most, tol = -1)
    
    a =al.Histograms(filtered)
    print("The absolute_treshold {}".format(a))
    a1 =al.Histograms(rel_filtered)
    print("The relative_treshold {}".format(a1))
    print(a)