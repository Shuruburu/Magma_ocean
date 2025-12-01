    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 13:34:06 2025

@author: root
"""
import Algorithms as al
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

def Find_eddit_in_the_file(line):
    if "atom_list" in line:
        keys = al.string_fragmentation(line, "'")
        return keys
    else:
        return None
def Find_the_elements(line, keys):
   i = 0
   for key in keys:
       if "{}_H".format(key) in line:
           while line[i]!= line[-1]:
               if line[i].isdigit() == True:
                   #load the data from the dictionary 
                   print(line[i], end= " ")
               i = i+1
   return True 

def Find_const(f, line, dictionary ):
    if"const_mix" in line:
        i=0
        while  line[i]!= line[-1] and line[i]!= "{":
            i = i+1
        if line[i]!= "{" :    
            #Writting in the file 
            f.seek(i)
            f.write(str(dictionary["const_mix"]))
            return False
        return True
    else:
        return True
def Find_P_T_profile(line):
    if "smg" in line:
        return
def Find_the_smg(f, line, dictionary):
    if "use_fix_sp_bot" in line:
        i =0
        while line[i]!= line[-1] and line[i]!= "{":
            i = i+1
        
        if line[i]!= "{" :
            f.seek(i)
            value = str(dictionary["const_mix"])
            
            return False
        return True
    else:
        return True
    
def Find_pressure(f,line):
    if"p_b" in line:
        i=0
        while line[i].isdigit():
            i =i+1
        f.seek(i)
        f.write()
        return True
    else:
        return False
def Find_dominat(f, line, dictionary):
    if"atm_base" in line:
        i=0
        while line[i]!= line[-1] and line[i]!= "[":
            i =i+1
        if line[i]=="[":
            f.seek(i)
            f.write("[{}]".format(dictionary["domiant"]))
            return False
        return True
    else:
        return True
##
#
def Write_In_the_file(filename, dictionary, pressure, number ,kzz, p_rad, T):
    #The fix condition  has been turned off due to the problem with negative solution in vulcan
    Flag = [False, True, True, True, True, True ]
    new_lines = []

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line and line[0] != '#':  # skip commented lines
            if Flag[0] and "use_fix_sp_bot" in line:
                Flag[0] = False
                value = str(dictionary["const_mix"])
                new_lines.append(f"use_fix_sp_bot = {value}\n")
                continue

            if Flag[1] and "const_mix =" in line:
                Flag[1] = False
                value = str(dictionary["const_mix"])
                new_lines.append(f"const_mix = {value}\n")
                continue

            if Flag[2] and "atm_base" in line:
                Flag[2] = False
                value = str(dictionary["dominant"])
                if value == "SO2":
                    new_lines.append(f"atm_base = 'CO2'\n")
                elif value == 'CH4':
                    new_lines.append(f"atm_base = 'CO2'\n")
                else:
                    new_lines.append(f"atm_base = '{value}'\n")
                continue
            if Flag[3] and "P_b" in line:
                Flag[2] = False
                value = pressure
                new_lines.append(f"P_b = {value}\n")
                continue
            if Flag[4] and "out_name" in line:
                Flag[4] = False
                value = number
                new_lines.append(f"out_name = 'Simulation_number_nr={value}_kzz={kzz}_p={p_rad}_Tempreture={T}.vul'\n")
                continue
 
        # Keep the original line if not modified
        new_lines.append(line)

    # Overwrite the file with new content
    with open(filename, "w") as f:
        f.writelines(new_lines)
def Eddit_Kzz(temp, presssure, eddy):
    filename = "/home/shurubura/Documents/VULCAN/atm/P_T_proffile.txt"
    with open(filename, "w") as f:
        f.write("#      (dyne/cm )    (K)     (cm2/s)\n " )
        f.write("   Pressure   Temp    Kzz\n")
        for i in range(temp.size):
            f.write("{}   {}   {}\n" .format(presssure[i]*1e6, temp[i] , eddy[i]))
            
    

    
def Load_the_excel(filename):
    df= pd.read_excel(filename)
    file = df.to_numpy(dtype=float)
    return file
        
