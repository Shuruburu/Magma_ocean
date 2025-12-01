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
#np.set_printoptions(threshold=sys.maxsize)

#NO H2_g
# find the expected atmosphere
# plotiing of the shit, some statistocs connected to plotting and the variance 
    

if __name__=="__main__":
    
    #guess = np.linspace(0, 10, 500)
    #Examples.Monte_Carlo()
    data = load.Load_the_pickle(load.pickle100T280())
    Plots =al.Plots(data)
    ratio =al.Element_ratios(data)
    
    preselected, most, preselected1 , error_bar= scripts.Selection()
    selected = al.Selection_of_mixing(preselected1)
    mixing_ratio =  al.Mixing_ratio(data, "volume_mixing_ratio" , selected)
    # Selecting pressures that have lower equalibrum pressure than 10
    
    to_be_writen_file =al.Add_the_domiant_species(mixing_ratio, most)
    scripts.Plot_the_groups(data, error_bar)
    
    Atmosphere = al.Atmosphere_model("In progress ",0, 1000 , data)
    Atmosphere.Adiabat_tempreture(700)
    pressure , tempretures , eddy = Atmosphere.Build_the_atmosphere()
    Plots.Plot_simple( tempretures, np.log10(pressure) , "tempreture" , "pressure" ,"P_T profile")
        
    
    
    to_be_writen_file =al.Add_the_domiant_species(mixing_ratio, most)
    if len(sys.argv) > 1:
        if sys.argv[1] =="Species_grid":
            #selected the atmobase 
            
            Atmosphere = al.Atmosphere_model("In progress ", int(sys.argv[2]), int(sys.argv[3]) , data)
            Atmosphere.Grid()
            #Missing the P_b file_edition while due to change of the pressure file 
            pressure , tempretures , eddy = Atmosphere.Build_the_atmosphere()
            load.Eddit_Kzz(tempretures, pressure, eddy)
            lines = load.Write_In_the_file("/home/shurubura/Documents/VULCAN/vulcan_cfg.py", 
                                              to_be_writen_file[int(sys.argv[2])], pressure[0], int(sys.argv[2]), int(sys.argv[3]) , 1 , 280)
            
        elif sys.argv[1] == "get_the_keys":
            for key in mixing_ratio.keys():
                print(key, end= " ")

        elif sys.argv[1] == "Information":
            print(most[int(sys.argv[2])], end = " ")
        
        elif sys.argv[1] =="Pressure_adiabat":
            Atmosphere = al.Atmosphere_model("In progress ", int(sys.argv[2]) , 1000, data)
            Atmosphere.Adiabat_pressure(int(sys.argv[3]))
            pressure , tempretures , eddy = Atmosphere.Build_the_atmosphere()
            Plots.Plot_simple(tempretures, np.log10(pressure), "temperatures", "pressure", "TPK_p_rad{}_280_1000".format(int(sys.argv[3])))
            load.Eddit_Kzz(tempretures, pressure, eddy)
            lines = load.Write_In_the_file("/home/shurubura/Documents/VULCAN/vulcan_cfg.py", 
                                              to_be_writen_file[int(sys.argv[2])], pressure[0], int(sys.argv[2]), 1000 ,int(sys.argv[3]) , 280)
            
        elif sys.argv[1] == "Tempreture_adiabat":
            Atmosphere = al.Atmosphere_model("In progress ", int(sys.argv[2]) ,1000 ,data)
            Atmosphere.Adiabat_tempreture(int(sys.argv[3]))
            pressure , tempretures , eddy = Atmosphere.Build_the_atmosphere()
            Plots.Plot_simple(tempretures, np.log10(pressure), "temperatures", "pressure", "TPK_p_rad_1_{}_1000".format(int(sys.argv[3])))
            load.Eddit_Kzz(tempretures, pressure, eddy)
            lines = load.Write_In_the_file("/home/shurubura/Documents/VULCAN/vulcan_cfg.py", 
                                              to_be_writen_file[int(sys.argv[2])], pressure[0], int(sys.argv[2]), 1000 , 1, int(sys.argv[3]))
                        