'''
############################################################################################################################
    Main script for extracting information and saving in .pkl files MULTIPLE rotation session (BMI Task: BMICursorRotErrorClamp in error_clamp_tasks.py)
        NOTE: This script EXCLUDES "error clamp" trials in analysis.
###########################################################################################################################    
    

Files Required to Run Script:

    Assumes that data is saved in the following heirarchy:
            Subject (e.g., Airport, Brazos)
                Degree Folder (e.g., 50, 90, neg50, neg90)
                    Folder for each session (e.g., 20220315, 20220316, 20220318)
                        3 Files/Folder ONLY: HDF, .pkl, .pkl


Analyses:
    Kalman gain 
    Neural Command (KY)
    Path length of cursor
    Variance Decomposition (factor analysis)


***Search '#!' for lines of code that need to be altered.***




@author: hanna
created on Sun Jan 23 12:18:06 2022
last updated: March 30, 2022
'''

#%%

import os
import glob
import pickle
import numpy as np 

''' #! Change directory to BMI PYTHON location.'''
os.chdir(r'C:\Users\hanna\OneDrive\Documents\bmi_python-bmi_developed')
import riglib

''' #! Change directory to CUSTOM FUNCTIONS location.'''
os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation')
from basic_fxns           import trialsNStuff
from generatePickles_fxns import extractKG, cursorPathLength, KY, varianceSharedPrivate

#Adjustable parameters.
subject = 'braz'
path = r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Data\Brazos'
dColors = {'BL': 'g', 'PE': 'darkgoldenrod'}
dTitles = {'BL': 'BL', 'PE': 'PE'}
targetDeg = np.arange(0,360,45)


#%% Kalman Gain

'###########################################################'
''' Kalman Gain (Baseline, Perturbation) for All Sessions '''
'############################################################'

KG = {}
date_list = []
deg_list = []

os.chdir(path)
degFolders = glob.glob('*')

for degFolder in degFolders:
    os.chdir(path+'/'+degFolder)
    dates = glob.glob('*')
    
    for date in dates:
        KG[date] = extractKG(date)
        deg_list.append(degFolder)
        date_list.append(date)
    
        print('Finished:', degFolder, date)

  
'''Saving values to .pkl'''     
os.chdir('../..')
obj_to_pickle = [deg_list, date_list, KG]
filename = subject+'_KG_BL_PE.pkl'
open_file = open(filename, "wb")
pickle.dump(obj_to_pickle, open_file)
open_file.close()


#%% Cursor Path Length

'###########################################################'
''' Cursor Path Length for All Sessions '''
'############################################################'

dist = {}
trial_num = {}
date_list = []
deg_list = []

os.chdir(path)
degFolders = glob.glob('*')

for degFolder in degFolders:
    os.chdir(path+'/'+degFolder)
    dates = glob.glob('*')
    
    for date in dates:
        dist[date], trial_num[date] = cursorPathLength(date)
        deg_list.append(degFolder)
        date_list.append(date)
        
        print('Finished:', degFolder, date)
        
'''Saving values to .pkl'''      
os.chdir('../..')
obj_to_pickle = [deg_list, date_list, trial_num, dist]
filename = subject + '_cursorPathLength.pkl'
open_file = open(filename, "wb")
pickle.dump(obj_to_pickle, open_file)
open_file.close()




#%% Variance - Private and Shared (factor analysis)

'###########################################################'
''' Shared and Private Variance (factor analysis) '''
''' NOTE: Must run KG first'''
'############################################################'

os.chdir(path)
os.chdir('..')
KG_file = glob.glob(subject+'*KG_BL_PE*')

'Open Kalman gain .pkl file'
open_file = open(KG_file[-1], "rb")
loaded = pickle.load(open_file)
KG = loaded[2]
open_file.close()


date_list = []
deg_list = []


nUnits = {} #Number of neurons used for decoder
KYx    = {} #neural command magnitude (x): Kalman gain (Kx) * spike counts (Y)
KYy    = {} #neural command magnitude (y): Kalman gain (Ky) * spike counts (Y)
dFA    = {} #dictionary that stores spike counts matrix for factor analysis
TS     = {} #total shared variance (factor analysis)
TP     = {} #total private variance (factor analysis)
ind    = {} #timepoint/index within a trial
x_pos  = {} #cursor x-position as saved in HDF
y_pos  = {} #cursor y-position as saved in HDF

os.chdir(path)
degFolders = glob.glob('*')

for degFolder in degFolders:
    os.chdir(path+'/'+degFolder)
    dates = glob.glob('*')
    
    for date in dates:
        KYx[date]    = {}
        KYy[date]    = {}
        dFA[date]    = {}
        TS[date]     = {}
        TP[date]     = {}
        ind[date]    = {}
        x_pos[date]  = {}
        y_pos[date]  = {}
        
        'Baseline'
        key = 'BL'
        num_neurons, FA, indT, x, y, NCx, NCy = KY(date, key, KG[date]['xBL'], KG[date]['yBL'])
        vTS, vTP = varianceSharedPrivate(date, num_neurons, 3, FA)
        
        nUnits[date] = num_neurons
        
        KYx[date][key]    = NCx
        KYy[date][key]    = NCy
        dFA[date][key]    = FA
        TS[date][key]     = vTS
        TP[date][key]     = vTP
        ind[date][key]    = indT
        x_pos[date][key]  = x
        y_pos[date][key]  = y

        
        'Perturbation'
        key = 'PE'
        _, FA, indT, x, y, NCx, NCy = KY(date, key, KG[date]['xPE'], KG[date]['yPE'])
        vTS, vTP = varianceSharedPrivate(date, num_neurons, 3, FA)
        
        KYx[date][key]    = NCx
        KYy[date][key]    = NCy
        dFA[date][key]    = FA
        TS[date][key]     = vTS
        TP[date][key]     = vTP
        ind[date][key]    = indT
        x_pos[date][key]  = x
        y_pos[date][key]  = y
        
        date_list.append(date)
        deg_list.append(degFolder)
        
        os.chdir('..')
        print('Finished:', degFolder, date)
        


obj_to_pickle = [deg_list, date_list, 
                 num_neurons, dFA, TP, x_pos, y_pos, KYx, KYy,
                 TS, TP]
                  

os.chdir(r'../..')
file_name = subject+'_KY_var.pkl'

open_file = open(file_name, "wb")
pickle.dump(obj_to_pickle, open_file)
open_file.close()

