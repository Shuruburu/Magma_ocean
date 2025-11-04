    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 13:34:06 2025

@author: root
"""

import pandas as pd
import pickle 
import numpy 
def Load_the_pickle(filename):
    df= pd.read_pickle(filename, )
    #with open(filename ,  "rb") as f:
      #  data = pickle.load(f)
    return df
def pickle100T280():
    name = "/home/shurubura/Documents/project/Data/t1e_280K(2).pkl"
    return name 

def phi100T280():
     name ="/home/shurubura/Documents/spyder/project/phi100/t1e_280K.xlsx"
     return name
def pickle100T1800():
    name = "/home/shurubura/Documents/spyder/project/phi100/t1e_1800K_with_solubility.pkl"
    return name 

def Save_data(data):
    with open('Integrated.pkl', 'wb') as f:
        pickle.dump(data, f)

    print("Data saved to 'data.pkl'")





def Write_In_the_file(filename):
    with open(filename ,  "r") as f :
        lines =f.readlines()
    return lines

def Load_the_excel(filename):
    df= pd.read_excel(filename)
    file = df.to_numpy(dtype=float)
    return file
        
