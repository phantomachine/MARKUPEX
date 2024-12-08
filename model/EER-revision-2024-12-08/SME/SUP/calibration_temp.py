# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 08:36:09 2024

@author: User
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import statsmodels.api as sm
import seaborn as sns
from mpi4py import MPI

import requests, zipfile, io
from pandas import ExcelWriter
from pandas import ExcelFile
from fredapi import Fred

import statistics as stats
from scipy import interpolate
from scipy.optimize import curve_fit

from itertools import product
import time

from datetime import date

import bcw_bj as klln

today = date.today()

# Instantiate Fred class as "fred", with personal own api_key 
# It can be registered through https://research.stlouisfed.org by searching api key
fred = Fred(api_key = 'cd6992c5ce004b0b4493fe2cf6c3fdab')

df = {}

#get data from Assets tab in Lucas-Nicolini excel file
data_LN_data = \
        pd.read_excel('Calibration_data.xlsx', 'data_consolidated')
        
# Store lists of data series and KEYs into dictionary df
df['Dates'] = data_LN_data['years'].tolist()
df['T-Bill Rate (3-month, annual)'] = (data_LN_data['tbills3m']/100.).tolist()
df['NewM1/GDP'] = (data_LN_data['new_m1']/100.).tolist()

# Firm
df['Markup'] = data_LN_data['markup'].tolist()
df['Markup dispersion'] = data_LN_data['markup dispersion'].tolist()

# inflation
inflation = fred.get_series('CPIAUCNS', frequency='a')
inflation = inflation.reset_index()
inflation.columns = ['Dates', 'CPI (seasonal)']
inflation = inflation.set_index(inflation['Dates'])
start_year = '1915'
stop_year = '2016' 
inflation = inflation['CPI (seasonal)']/inflation['CPI (seasonal)'].shift(1) - 1

df['Inflation (CPI, raw)'] = inflation.loc[start_year+'-01-01':stop_year+'-01-01']

obslength = []
keys = []

for list_idx, list_key in enumerate(df):
    list_length = len(df[list_key])
    print(list_idx, list_key, "> Observations:", list_length)
    obslength.append(list_length)
    keys.append(list_key)
obslength_min = min(obslength)
print("Shortest data series is %s with data length of %i" %(keys[obslength==obslength_min], obslength_min))

# Truncate longer series to have the same length of: obslength_min
for list_idx, list_key in enumerate(df):
    df[list_key] = df[list_key][0:obslength_min]
    
# Convert df to Pandas dataframe
d = pd.DataFrame(df)
d = d.set_index(d['Dates'])   

# Truncate data window
date_truncate_start = '1980' 
date_truncate_stop = '2007'

dcut = d.loc[date_truncate_start:date_truncate_stop]
        
# Truncated sample (1980-2007) stats
dcut_mean = dcut.mean().to_frame()
dcut_mean.columns = ["sample mean"]
dcut_mean

T_bill_mean = 0.058154
τ_mean = 0.038494
#Use Fisher relation to back out the discount factor
β_data= (1.0+τ_mean)/(1.0+T_bill_mean)
print(β_data)

def func(x, a, b, c):
    return a*np.exp(-b * x) + c 

def money_func_fit(x, y):
    data = np.asarray(sorted(zip(x,y)))
    i_set = np.linspace(data[:,0].min(), data[:,0].max())
    popt, pcov = curve_fit(func, x, y, bounds=(0.001, 10.0))
    fitted_func = func(i_set, *popt)
    return fitted_func

def markup_func_fit(x,y):
    data = np.asarray(sorted(zip(x,y)))
    i_set = np.linspace(data[:,0].min(), data[:,0].max())
    popt, pcov = curve_fit(func, x, y, bounds=(0.001, 10.0))
    fitted_func = func(i_set, *popt)
    return fitted_func

def visualization(x = dcut['T-Bill Rate (3-month, annual)'].tolist(),\
                  y = dcut['Markup'].tolist(),\
                  x_label = 'T-Bill Rate (3-month, annual)',\
                  y_label = 'Markup',\
                  model_x = np.linspace(0.0,0.0,1),\
                  model_y = np.linspace(0.0,0.0,1)
                 ):
    data = np.asarray(sorted(zip(x,y)))
   
    i_set = np.linspace(data[:,0].min(), data[:,0].max())
    if y_label == 'NewM1/GDP':
        popt, pcov = curve_fit(func, x, y, bounds=(0.001, 10.0))
    else:
        popt, pcov = curve_fit(func, x, y, bounds=(0.001, 10.0))
    fitted_func = func(i_set, *popt)
    if y_label == 'NewM1/GDP':
        plt.scatter(x, y, label='Data: 1980-2007')
        plt.plot(i_set, fitted_func, '--', color='r', label='Fitted spline')
        plt.plot(model_x, model_y, '-g', label="Model")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(0.0, 0.16)
        plt.ylim(0.1, 0.4)
        plt.legend()
        plt.show()
    else:
        plt.scatter(x, y, label='Data: 1980-2007')
        plt.plot(i_set, fitted_func, '--', color='r', label='Fitted spline')
        plt.plot(model_x, model_y, '-g', label="Model")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(0.0, 0.16)
        plt.ylim(1.0, 1.6)
        plt.legend()
        plt.show()    
    return fitted_func


σ_DM_range = np.linspace(0.3, 0.4, 2)  
Ubar_CM_range = np.linspace(1.89, 1.905, 2)
α_1_range = np.linspace(0.4,0.5, 2)


#create a for-loop parameter value table
KLLN_class = np.array( [klln.baseline_mod(β=β_data, σ_CM=1.0, σ_DM=i, Ubar_CM=n, α_1 =j) \
                            for (i, j, n) in product(σ_DM_range,\
                                                        α_1_range,\
                                                        Ubar_CM_range)]) 
    

print(KLLN_class.size)


def L(M1_GDP_target):
    """distance function between model simulated stat vs. spline-fitted data target"""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Distribute KLLN_class among processes
    local_size = len(KLLN_class) // size
    start_idx = rank * local_size
    end_idx = start_idx + local_size if rank != size-1 else len(KLLN_class)
    local_KLLN_class = KLLN_class[start_idx:end_idx]
    
    # Local calculations
    local_L_diff = np.zeros(len(local_KLLN_class))
    
    for idx, x in enumerate(local_KLLN_class):
        # Calculate for each local class instance
        z_guess, i_guess = 0.4, 0.01
        result = local_KLLN_class[idx].SME_stat(z_guess, i_guess)
        
        mpy_grid = result['stat_grid']['mpy_star']
        i_FFR_grid = result['stat_grid']['FFR']
        
        w_mpy = 1.0/(stats.variance(M1_GDP_target))
        diff_M1_GDP = np.abs(mpy_grid - M1_GDP_target)
        
        L_diff_func = np.sum(w_mpy*(diff_M1_GDP)**2.0)
        local_L_diff[idx] = L_diff_func.min()
    
    # Gather all results to rank 0
    if rank == 0:
        L_diff = np.zeros(len(KLLN_class))
    else:
        L_diff = None
        
    comm.Gatherv(local_L_diff, [L_diff, (np.array([local_size]*size), None), MPI.DOUBLE], root=0)
    
    if rank == 0:
        # Process final results in rank 0
        L_diff_min = L_diff[L_diff.argmin()]
        KLLN_model = KLLN_class[L_diff.argmin()]
        model_result = KLLN_model.SME_stat(z_guess, i_guess)
        
        M1_GDP_star = model_result['stat_grid']['mpy_star']
        τ_mean = 0.038
        z_mean, i_mean = KLLN_model.solve_z_i(z_guess, i_guess, τ_mean)  
        markup_star = KLLN_model.markup_func(z_mean, i_mean, τ_mean) 
        i_policy_grid = model_result['stat_grid']['FFR']
        
        # Get parameter values
        σ_DM = KLLN_model.σ_DM
        σ_CM = KLLN_model.σ_CM
        α_1 = KLLN_model.α_1
        Ubar_CM = KLLN_model.Ubar_CM    
        
        return M1_GDP_star, markup_star, i_policy_grid, σ_DM, σ_CM,α_1, Ubar_CM, L_diff_min, KLLN_model
    
    return None

M1_GDP_fit_target =visualization(x = dcut['T-Bill Rate (3-month, annual)'].tolist(),\
                  y = dcut['NewM1/GDP'].tolist(),\
                  x_label = 'T-Bill Rate (3-month, annual)',\
                  y_label = 'NewM1/GDP',\
                  model_x = np.linspace(0.0,0.0,8),\
                  model_y = np.linspace(0.0,0.0,8))
    
markup_fit_target=visualization(x = dcut['T-Bill Rate (3-month, annual)'].tolist(),\
                  y = dcut['Markup'].tolist(),\
                  x_label = 'T-Bill Rate (3-month, annual)',\
                  y_label = 'Markup',\
                  model_x = np.linspace(0.0,0.0,1),\
                  model_y = np.linspace(0.0,0.0,1))    

markup_target = markup_fit_target
M1_GDP_target = M1_GDP_fit_target

tic = time.time()
M1_GDP_star, markup_star, i_policy_grid, σ_DM, σ_CM, λ, Ubar_CM,\
            L_diff_min, KLLN_model = L(M1_GDP_target)           
toc = time.time()-tic
print(toc, "seconds")

rowlabels = ["$\sigma_{DM}$", "B (Ubar_CM)",\
             "$λ$"]

model_para = [ σ_DM, Ubar_CM, λ]
model_para_table = { "Calibrated parameters": rowlabels,
                     "Model (Avg)": model_para                
                   }
# Create dataframe
df_model_parameter = pd.DataFrame(data=model_para_table)

print(markup_star)