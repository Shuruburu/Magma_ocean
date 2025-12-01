#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:50:10 2025

@author: root
"""
#
Earth_mass = 5.97e24
#

from scipy.stats import norm
import matplotlib.pyplot as plt
import logging
import numpy as np
import matplotlib.patches as mpatches




# List of species used in the simulation of Traphiest 1e atmosphere


def Get_species_elements(data, start_key=None, start_index=0, max_elements=6):
    """
    Extracts keys, values (as numpy arrays), and columns from a dictionary.

    Parameters:
    - data: dict of {key: pandas.DataFrame or Series}
    - start_key: key name to start from (overrides start_index if given)
    - start_index: index to start from if start_key is None
    - max_elements: maximum number of elements to take (None = all)

    Returns:
    - keys: list of keys from data
    - columns: list of columns of each value
    - values: list of numpy arrays from each value
    """
    keys = []
    values = []
    columns = []

    # Convert dict to list of items for slicing
    items = list(data.items())

    # Determine starting index
    if start_key is not None:
        try:
            start_index = [k for k, v in items].index(start_key)
        except ValueError:
            raise ValueError(f"Key '{start_key}' not found in the dictionary.")

    # Slice items according to start_index and max_elements
    if max_elements is not None:
        sliced_items = items[start_index:start_index + max_elements]
    else:
        sliced_items = items[start_index:]

    # Collect keys, values, and columns
    for key, value in sliced_items:
        keys.append(key)
        values.append(value.to_numpy())
        if hasattr(value, 'columns'):  # DataFrame
            columns.append(list(value.columns))
        else:  # Series
            columns.append([value.name] if value.name else ['value'])

    return keys, columns, values
def Find_index_in_column(columns,name):
    for i in range(len(columns)):
        if name == columns[i]:
            return i
    return -1

def Plot_specific_column(data , column, x_axis,  y_axis):
    plt.scatter(data[x_axis][column].to_numpy()/Earth_mass, data[y_axis][column].to_numpy()/Earth_mass,label  = column )
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(column)
    plt.savefig("/home/shurubura/Documents/project/Magma/plots/{}".format(column))
    plt.show()



def to_python_number(x):
    """
    Convert a NumPy numeric type to a native Python int or float.
    """
    if isinstance(x, (np.integer, int)):
        return int(x)
    elif isinstance(x, (np.floating, float)):
        return float(x)
    else:
        raise TypeError(f"Cannot convert type {type(x)} to Python number.")
        
        
        
#TO DO finishing writting this shit 
def Integrate_all(data, keys):
    """
    Computes the average value of specified columns for each key in a dataset,
    returning a dictionary with the same top-level key names as the input `data`.

    Parameters:
        data (dict): A dictionary where each key maps to a DataFrame or similar data structure.
        keys (list): A list of keys (subset of data.keys()) to process.
        columns (list): A list of column names whose mean values will be computed.

    Returns:
        dict: A dictionary in the format:
              {
                  key1: {column1: avg_value, column2: avg_value, ...},
                  key2: {column1: avg_value, column2: avg_value, ...},
                  ...
              }
    """

    results = {}  # Dictionary to store results per key

    # Iterate through each key
    for key in keys:
        if key not in data:
            continue  # Skip keys that aren't in the data dictionary

        results[key] = {}  # Create a nested dictionary for this key

        # Loop through each requested column
        for column in data[key].keys():
            if column not in data[key].columns:
                continue  # Skip columns that don't exist in this DataFrame

            # Extract the column data as a NumPy array
            column_data = data[key][column].to_numpy()

            # Compute the mean value safely
            avg = sum(column_data) / column_data.size if column_data.size > 0 else float('nan')

            # Store the result
            results[key][column] = to_python_number(avg)

    return results



        


def Integrate_Keys(data, keys, column, max1= False):
    """
    Computes the average value of a specified column for each key in a dataset.

    Parameters:
        data (dict): A dictionary where each key maps to a DataFrame or similar data structure.
        keys (list): A list of keys to iterate through in 'data'.
        column (str): The column name whose mean value will be computed.

    Returns:
        list: A list of [key, average_value] pairs.
    """

    values = []  # Initialize an empty list to store [key, average_value] pairs
    max_value = 0 
    # Loop through each key in the provided list
    for i , key in enumerate(keys):
        # Extract the column as a NumPy array for the current key's data
        column_data = data[key][column].to_numpy()

        # Compute the average (mean) value of the column
        temp = sum(column_data) / column_data.size
        if max1  and max_value < temp:
            max_value = temp
            max_index = i
        # Append the result as a pair [key, average_value] to the list
        values.append([key, temp])

    # Return the complete list of results
    if max1:
        return values , max_index
    else:
        return values
def classify_the_atmosphere(data, keys, column):
    """
    Find, for each row (index j), which dataset (key) has the largest value
    in the specified column.

    Parameters
    ----------
    data : dict
        A dictionary where each value is a pandas DataFrame (or Series).
        Example: data["sensor1"]["temperature"] → column of values.
    keys : list
        List (or array) of keys corresponding to the items in 'data'.
    column : str
        Name of the column to compare between datasets.

    Returns
    -------
    list
        List of indices (corresponding to 'keys') indicating which dataset
        had the highest value in each row.
    """

    max_values = []   # Stores the maximum value found for each row (optional info)
    max_indices = []  # Stores which key had the maximum value for each row
    # Loop through each row index (0 → 9999)
    for j in range(10000):
        current_max = float('-inf')  # Reset current maximum for this row
        current_max_index = None     # Reset index of dataset with max value

        # Loop through each dataset
        for i in range(len(keys)):
            # Extract the j-th value in the given column for this dataset
            value = data[keys[i]][column].to_numpy()[j]

            # If this value is larger than the current max, update
            if value >= current_max:
                current_max = value
                current_max_index = keys[i]

        # Store results for this row
        max_values.append(current_max)
        max_indices.append(current_max_index)

    # Return the indices of the datasets with the highest value per row
    return max_indices
def Compare_arrays(list1 , list2):
    for i , element in enumerate((list1)):
        if element!= list2[i]:
            print(" Here {}".format(i))
    return

def Create_mash_list(list1, list2):
    list2d  = [[]]
    for i in range(len(list1)):
        
        list2d.append([list1[i] , list2[i]])
        
    return list2d

def EM_algorithm(X, n_classes=4, n_iter=200, tol=1e-6, random_state=0, Guess=0):
    np.random.seed(random_state)
    n, d = X.shape

    # --- Initialization (no KMeans) ---
    # Choose random data points as initial means
    mu = np.random.rand(5, 2)
    
    # Initialize sigma as global std
    sigma = np.ones((n_classes, d)) * X.std(axis=0)
    pi = np.ones(n_classes) / n_classes
    eps = 1e-12
    prev_ll = -np.inf

    for it in range(n_iter):
        # --- E-step ---
        log_px_given_k = np.zeros((n, n_classes))
        for k in range(n_classes):
            # sum log pdfs for independent dimensions
            for j in range(d):
                log_px_given_k[:, k] += np.log(norm.pdf(X[:, j], mu[k, j], sigma[k, j]) + eps)
        
        log_num = np.log(pi + eps)[None, :] + log_px_given_k

        # log-sum-exp for normalization
        a_max = np.max(log_num, axis=1, keepdims=True)
        log_den = a_max + np.log(np.sum(np.exp(log_num - a_max), axis=1, keepdims=True))
        log_resp = log_num - log_den
        gamma = np.exp(log_resp)

        # --- M-step ---
        N_k = gamma.sum(axis=0)
        pi = N_k / n
        mu = (gamma.T @ X) / N_k[:, None]

        # Update variances
        sigma = np.zeros((n_classes, d))
        for k in range(n_classes):
            diff = X - mu[k]
            sigma[k] = np.sqrt(np.sum(gamma[:, k][:, None] * diff**2, axis=0) / (N_k[k] + eps))
            sigma[k] = np.maximum(sigma[k], 1e-3)  # prevent collapse

        # --- Log-likelihood ---
        ll = np.sum(log_den)
        if abs(ll - prev_ll) < tol:
            print(f"Converged at iteration {it}")
            break
        prev_ll = ll

    return mu, sigma, pi

def select_max(prop):
    max_value = 0
    max_index = np.empty(prop.shape[0])
    for i in range(prop.shape[0]):
        for k in range(prop.shape[1]):
            if prop[i][k] >= max_value:
                max_value = prop[i][k]
                max_index[i] = k
        max_value = 0
    return max_index
def Give_the_label(X_new, sigma, mu, pi):
    """
    X_new: (n_points, 2)
    mu: (n_classes, 2)
    sigma: (n_classes, 2)
    pi: (n_classes,)
    Returns: (n_points, n_classes) posterior probabilities
    """
    n_points = X_new.shape[0]
    n_classes = mu.shape[0]
    probs = np.zeros((n_points, n_classes))
    
    for k in range(n_classes):
        # compute P(X_new | z=k) under Gaussian Naive Bayes
        p = np.ones(n_points)
        for j in range(2):  # for each feature
            p *= norm.pdf(X_new[:, j], mu[k, j], sigma[k, j])
        probs[:, k] = pi[k] * p  # multiply by class prior
    
    # normalize to get posterior probabilities
    probs /= (probs.sum(axis=1, keepdims=True) + 1e-12)
    
    return select_max(probs)

class Plots:
    def __init__(self, data):
        self.color_map = {
            0: "orange",
        1: "green",
        2: "blue",
        3: "red",
        4: "yellow",
        5: "purple",
        6: "pink",
        7: "brown",
        8: "black",
        9: "white"
            }   
        self.data = data
        self.save = "/home/shurubura/Documents/project/Magma/plots/"
    def Mask(self, filtered_data):
        # Ensure all elements are boolean arrays
        bool_lists = [[bool(x) for x in lst] for lst in filtered_data]
    
        # Elementwise AND across all arrays
        mask = [all(values) for values in zip(*bool_lists)]
    
        return mask
    def Plot_filtered(self, key1, key2, column, filtered_data, labels):
        raw_x = self.data[key1][column]
        raw_y = self.data[key2][column]
        raw_labels = np.array( labels)
        
        mask = self.Mask(filtered_data)
    
        x = raw_x[mask]
        y = raw_y[mask]
        labels = raw_labels[mask]
        # Assign colors
        color = [self.color_map[item] for item in labels]
        #print(color)
        # Scatter plot
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(np.log10(x), np.log10(y), c=color)
    
        plt.xlabel(f"{key1} (log10)")
        plt.ylabel(f"{key2} (log10)")
        plt.title(f"{key1} vs {key2} in {column}")
        plt.grid(True, which='both', ls='--')
    
        # Create legend
        unique_labels = np.unique(labels)
        patches = [mpatches.Patch(color=self.color_map[label], label=str(label)) for label in unique_labels]
        plt.legend(handles=patches, title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    
        # Save and show
        plt.tight_layout()
        plt.savefig("/home/shurubura/Documents/project/Magma/plots/{} vs {}. with 3 element.png".format(key1, key2), dpi=300)
        plt.show()
    def Plot_general_log(self, key1, key2 , column, column1):
        x = self.data[key1][column]
        y = self.data[key2][column1]
        
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(np.log10(x), np.log10(y))
    
        plt.xlabel(f"{key1} (log10)")
        plt.ylabel(f"{key2} (log10)")
        plt.title(f"{key1} vs {key2} in {column}")
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig("/home/shurubura/Documents/project/Magma/plots/{} vs {}. with 3 element.png".format(key1, key2), dpi=300)
        plt.show()
    def Histograms(self, bins, name  = " "):
        counts = bins.values()
        labels = bins.keys()
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(counts)), counts, color='skyblue')
        plt.xticks(range(len(labels)), labels, rotation=90)  # rotate labels for readability
        plt.xlabel('Gas Combinations')
        plt.ylabel('Count')
        plt.title('Histogram of {} Gas Mixtures'.format(name))
        plt.savefig("/home/shurubura/Documents/project/Magma/plots/Histogram_{}.png".format(name))
        plt.tight_layout()  # adjust layout to prevent label cutoff
        plt.show()
    def Histograms_list(self, bins, name, key, most, error_bar):

        counts = [np.log10(self.data[keys]["volume_mixing_ratio"][key]) for keys in bins]

        # ---- Convert error bars to log scale ----
        yerror = []
        if name in error_bar:
            for species in bins:
                value = self.data[species]["volume_mixing_ratio"][key]
                err = error_bar[name][f"M2_specie_{species}"]

                # Avoid negative or zero values
                if value > 0:
                    upper_log = np.log10(value + err) - np.log10(value)
                else:
                    upper_log = 0

                yerror.append(upper_log)

        labels = bins
        plt.figure(figsize=(12, 6))

        colors = [self.color_map[k] for k in range(len(bins))]

        plt.bar(
            range(len(counts)),
            counts,
            color=colors,
            yerr=yerror if len(yerror) > 0 else None,
            capsize=5
        )

        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.xlabel('Gas Combinations')
        plt.ylabel('The ratio between the most abundant gas')
        plt.title(f'Histogram of {name} Gas Mixtures nr {key}')

        plt.gca().invert_yaxis()  # keep the inverted axis

        plt.tight_layout()
        plt.savefig(f"/home/shurubura/Documents/project/Magma/plots/Histogram_{name}.png")
        plt.show()
        
    def Plot_simple(self, x ,y , xlabel ,ylabel, name, label = "P_T_profile" ):
        plt.plot(x , y , label = label, color = "red")
        plt.title(name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(f"/home/shurubura/Documents/project/Magma/plots/{name}.png")
        #plt.show()
        
def Plot_the_Key_data(data, key,  x_axis, y_axis, log =False ):
    if log == False:
        plt.scatter(data[key][x_axis].to_numpy(), data[key][y_axis].to_numpy(),label  = key )
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(key)
        plt.savefig("/home/shurubura/Documents/project/Magma/plots")
        plt.show()
    else:
        plt.scatter(np.log10(data[key][x_axis].to_numpy()), np.log10(data[key][y_axis].to_numpy()),label  = key )
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title("{} vs {}".format(x_axis ,y_axis ))
        plt.savefig("/home/shurubura/Documents/project/Magma/plots")
        plt.grid(True, which='both', ls='--')
        plt.show()

    
    
def Integrate(data, keys, column):
    temp = np.zeros(data[keys[0]][column].to_numpy().shape)
    for i, key in enumerate(keys):
        temp  = temp + data[key][column].to_numpy()
    return temp


def Get_a_specific_property(columns, name, values):
    index =Find_index_in_column(columns, name)
    values = 0
def Filter_out_data(data, column,tol = -10):
    keys, columns, values =Get_species_elements(data, start_index=0, max_elements=14)
    filter_data = {}
    
    list1 = []
    for i,  element  in enumerate(keys):
        for j, values in enumerate(data[element][column]):
            
            
            if np.log10(values) > tol:
                
                list1.append(1)
            else:
                list1.append(0)
        filter_data[element] = list1
        list1 = []
    return filter_data
def Filter_out_data_relative(data, column,dominant,tol = 2/1000):
    keys, columns, values =Get_species_elements(data, start_index=0, max_elements=14)
    filter_data = {}
    list1 = []
    for i,  element  in enumerate(keys):
        for j, values in enumerate(data[element][column]):
            if values> tol * data[dominant[j]][column][j]:
                list1.append(1)
            else:
                list1.append(0)
        filter_data[element] = list1
        list1 = []
    return filter_data

def Histograms(filtered_data):
    my_list = []
    counter = {}
    for j in range(10000):
        for element in filtered_data.keys():
            if filtered_data[element][j] == 1:
               my_list.append(element)
        item = str(my_list)
        my_list = []
        if item in counter:
            counter[item] += 1
        else:
            counter[item] = 1

    return counter
def Seleciton(filtered_data):
    my_list = []
    main_list  = []
    for j in range(10000):
        
        for element in filtered_data.keys():
            if filtered_data[element][j] == 1:
               my_list.append(element)
        item = str(my_list)
        my_list = []
        main_list.append([j, item])

    return main_list
def Select_unique(selected_data, not_rel_filtered_data, selection_of_pressure):

    unique = {}
    unique_output = {}
    unique_abs = {}
    for n, elements in enumerate(selected_data):
        if elements[1] in unique or elements[0] in selection_of_pressure.keys():
                continue
        else:
                unique[elements[1]]= elements[0]
                unique_output[elements[0]] = elements[1]
                unique_abs[elements[0]] = not_rel_filtered_data[n][1]
    return unique_output, unique_abs








def Prepare_to_write(data, selected):
    dictionary = {}
    for elements in selected.keys():
        #Get all the dictionary data that I need 
        dictionary[elements] =data[:][:][selected[elements]]
        #
    return dictionary
def Element_ratios(data):
    keys, columns, values =Get_species_elements(data, start_index=18, max_elements=6)
    ratio = {}
    for elements in keys:
        ratio["{}/element_H".format(elements)]=data[elements]["total_number"] / data["element_H"]["total_number"] 
    return ratio



def Mixing_ratio(data, column ,preselected):
    """Mixing_ratio_dictionary, if new species should be added please add here"""
    ratio = {} 
    for elements in preselected:
        ratio[elements[1]] = {}
        ratio[elements[1]]["const_mix"] = {}
        for species in elements[0]:
            if species == "ClH_g":
                continue
            elif species  == "O2S_g":
                value = data[species][column][elements[1]]    
                ratio[elements[1]]["const_mix"]["SO2"] = float(value)
            elif species == "H3N_g":
                value = data[species][column][elements[1]]    
                ratio[elements[1]]["const_mix"]["NH3"] = float(value)
            else:
                value = data[species][column][elements[1]]    
                ratio[elements[1]]["const_mix"][Adjusted_key(species)] = float(value)
    return ratio

def string_fragmentation(key, indicator):
    fragments = []
    start = None  # track position of opening quote
    
    for i, char in enumerate(key):
        if char == indicator:
            if start is None:
                # mark where the quoted part starts
                start = i+1
            else:
                # closing quote found → extract substring
                fragments.append(key[start:i])
                start = None  # reset for next pair
    return fragments

def Adjusted_key(key):
    if "_g" in key:
        key =key.replace("_g" ,"")
    return key
def Selection_of_mixing(preselected_data):
    list1 =[]
    for key in preselected_data.keys():
        list1.append([string_fragmentation(preselected_data[key], "'") ,  key])
    
    return list1
    
def Add_the_domiant_species(mixing_ratio, dominant):
    for key in mixing_ratio.keys():
        most =dominant[key]
        if most == "H3N_g":    
            mixing_ratio[key]["dominant"] = "NH3"
        elif most == "O2S_g":
            mixing_ratio[key]["dominant"] = "SO2"
        else:
            mixing_ratio[key]["dominant"] = Adjusted_key(dominant[key])
    return mixing_ratio
def Atm_base(mixing_ratio):
    list1 =  []
    for key in mixing_ratio.keys():
        dist  = mixing_ratio[key]["const_mix"]
        for species in dist.keys():
            #
            continue
def Pressure_filter(data, column , tol = 50):        
    filtered_data = {}
    for j, values in enumerate(data["atmosphere"][column]):
        if values > tol:
            filtered_data[j] = values
        else:
            continue
    
    filtered_data[j] = values
    return filtered_data


class Atmosphere_model :
    def __init__(self, path, number, kzz, data):
        self.config = path
        self.number = number
        self.kzz = kzz
        self.data = data 
        self.p_surf    = data['atmosphere']['pressure'][number]
        logp_surf = np.log10(self.p_surf)
        self.logp      = np.linspace(logp_surf,-8, 100)
        self. pressures = 10**self.logp
        self.T_eq = self.data['atmosphere']['temperature'][number]
    def Adiabat_pressure(self, p_rad =1):
    
        #K_z  =  10000
        
        self.tempretures=  self.T_eq*np.ones(self.pressures.size)
        
        
        # radiating pressure level (where the troposphere meets the stratosphere): units bar

        T_strat = self.T_eq #* 2**(-1/2) # the skin temperature approximation
        R_gas = 8314.5 # gas constant: units (m/s^2)/K  <---- different unit than the standard one 
        mmw   = self.data['atmosphere']['molar_mass'][self.number]*1e3 # mean molecular weight: units g/mol
        cp    = 820 # heat capacity, the value here is for CO2: units SI
        K_adiabat = R_gas / (mmw*cp) # the 'adiabatic index'

        T_adia_from_Prad = np.ones_like(self.logp)
        for i, p in enumerate(self.pressures):
            T_adia_from_Prad[i] = T_strat*(p/p_rad)**(K_adiabat)
            if T_adia_from_Prad[i] < T_strat: T_adia_from_Prad[i] = T_strat
        
        
        self.temperatures = T_adia_from_Prad
        self.eddy = np.ones(self.pressures.size)*self.kzz
        #z = np.arange(0, pressures.size)
    def Adiabat_tempreture(self, ground_tempreture = 900):
        self.tempretures = self.T_eq*np.ones(self.pressures.size)
        T_surface = ground_tempreture  
        
        
        R_gas = 8314.5
        mmw   = self.data['atmosphere']['molar_mass'][self.number]*1e3 # mean molecular weight: units g/mol
        cp    = 820 # heat capacity, the value here is for CO2: units SI
        K_adiabat = R_gas / (mmw*cp) # the 'adiabatic index'
        p_rad = self.p_surf*(self.T_eq/T_surface)**(1/K_adiabat)
        T_adia_from_Prad = np.ones_like(self.logp)
        for i, p in enumerate(self.pressures):
            T_adia_from_Prad[i] = self.T_eq*(p/p_rad)**(K_adiabat)
            if T_adia_from_Prad[i] < self.T_eq: T_adia_from_Prad[i] = self.T_eq
            
    
        self.temperatures = T_adia_from_Prad
        self.eddy = np.ones(self.pressures.size)*self.kzz
        
    def Grid(self ):
        # first, make pressure grid
        K_z  = self.kzz
        pressures = 10**self.logp
        self.temperatures=  self.T_eq*np.ones(pressures.size)
        self.eddy = np.ones(pressures.size)*K_z
        z = np.arange(0, pressures.size)
    
    def Eddy_simulate(self):
        ##################
        # Eddy diffusion #
        ################## Work in progress
        # Make a different eddy diffusion profile
        # Constant
        deep_Kzz = self.kzz
        Edd_const = np.ones_like(self.logp) * deep_Kzz

        # Deep Kzz then increase as a power law in pressure to a maximum
        p_trans = 1
        slope   = -0.4 # in logp units
        max_Kzz = 1e7

        Edd_power_law = np.ones_like(self.logp)
        for i, p in enumerate(self.pressures):
            Edd_power_law[i] = deep_Kzz*(p/p_trans)**(slope)
            if Edd_power_law[i] < deep_Kzz: Edd_power_law[i] = deep_Kzz
            if Edd_power_law[i] > max_Kzz:  Edd_power_law[i] = max_Kzz

        # Deep Kzz then drop to a lower value before increasing as a power law in pressure
        trans_Kzz = 1e3

        Edd_reduce = np.ones_like(self.logp)
        for i, p in enumerate(self.pressures):
            Edd_reduce[i] = trans_Kzz*(p/p_trans)**(slope)
            if p > p_trans: Edd_reduce[i] = deep_Kzz
            if Edd_reduce[i] > max_Kzz:  Edd_reduce[i] = max_Kzz

    def Build_the_atmosphere(self):
        return self.pressures, self.temperatures, self.eddy

""" TO DO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
# could add to the class if needed 
def Error_bars(data, counter, column, selected_rel, selected_abs):
    error_bar = {}

    for keys in counter.keys():
        for element in selected_rel:

            if element[1] == keys:

                species = string_fragmentation(selected_abs[element[0]][1], "'")

                for specie in species:
                    value = float(data[specie][column][element[0]])

                    if keys not in error_bar:
                        error_bar[keys] = {}

                    mean_key = f"mean_specie_{specie}"
                    count_key = f"nr_elements_{specie}"
                    m2_key = f"M2_specie_{specie}"   # running sum of squared deviations

                    # First occurrence of this species
                    if mean_key not in error_bar[keys]:
                        error_bar[keys][mean_key] = float(value)
                        error_bar[keys][count_key] = float(1.0)
                        error_bar[keys][m2_key] = float(0.0)
                    
                    else:
                        old_mean = float(error_bar[keys][mean_key])
                        old_count = float(error_bar[keys][count_key])
                        old_m2 = float(error_bar[keys][m2_key])

                        new_count = old_count + 1.0
                        delta = value - old_mean
                        new_mean = old_mean + delta / new_count
                        delta2 = value - new_mean

                        new_m2 = old_m2 + delta * delta2

                        error_bar[keys][mean_key] = new_mean
                        error_bar[keys][count_key] = new_count
                        error_bar[keys][m2_key] = new_m2

    return error_bar
        