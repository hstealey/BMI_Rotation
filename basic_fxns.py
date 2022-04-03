# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:15:20 2022

@author: hanna
"""

import os
import glob
import tables
import numpy as np
import pandas as pd
from   matplotlib import cm
import matplotlib.pyplot as plt
from   collections import Counter
from   sklearn.linear_model import LinearRegression
from   scipy import stats
from   factor_analyzer import FactorAnalyzer
import warnings
import pickle
import seaborn as sb
from   numpy.linalg import norm
import statsmodels.api as sm
from   statsmodels.formula.api import ols
from   statsmodels.stats.multicomp import pairwise_tukeyhsd
from   numpy import concatenate as ct 
from   datetime import datetime
from   csv import DictWriter

os.chdir(r'C:\Users\hanna\OneDrive\Documents\bmi_python-bmi_developed')
import riglib

#Custom functions
os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation')
from tuningCurve_fxns import performFA, spikeCount, tuningCurve, tuningCurveFA, targetDirectionFR#, tuningCurveTrialAverage, targetDirectionTwoBins
from behavior_fxns    import basicCursorTimes, plotTimes


def trialsNStuff(date, ABC=False):
    
    
    try:
        os.chdir(date)
    except Exception:
        pass
    
    #Adjustable parameters
    c = ['green', 'darkgoldenrod', 'blue', 'black']
    titles = ['Baseline', 'Pert (Early)', 'Pert (Late)', 'Washout']
    num_bins = 8 #for tuning curve
    num_factors = 10 #for factor analysis
    warnings.filterwarnings('ignore')
    plt.rcParams.update({'font.sans-serif': 'Arial'})


    hdf = tables.open_file(glob.glob('*.hdf')[0])
    decoder_files = glob.glob('*.pkl')
    decoder_filename = decoder_files[0]
    KG_picklename = decoder_files[1]


    ''' Determine units used in decoder. '''
    kf = pickle.load(open(decoder_filename, 'rb'), encoding='latin1')
    u = kf.units
    
    unitCode = {2: 'a',
                4: 'b',
                8: 'c',
                16: 'd'}
    
    units = [str(u[i,0])+unitCode[u[i,1]] for i in range(len(u))]
    
    
    '''Load KG values for task.'''
    f = open(KG_picklename, 'rb')
    KG_stored = []
    while True:
        try:
            KG_stored.append(pickle.load(f))
        except Exception:
            break
    f.close()
    
    
    '''Parse HDF file.'''
    msg = hdf.root.task_msgs[:]['msg'] 
    ind = hdf.root.task_msgs[:]['time']
    
    deg = hdf.root.task._v_attrs['rot_angle_deg']
    
    pert = hdf.root.task[:]['pert'][:,0]
    error_clamp = hdf.root.task[:]['error_clamp'][:,0]
    
    reward_msg = np.where(msg == b'reward')[0]
    holdCenter_msg = np.subtract(reward_msg,5)
    
    reward_ind = ind[reward_msg]
    holdCenter_ind = ind[holdCenter_msg]
    
    block_type = hdf.root.task[:]['block_type'][:,0]
    
    df = pd.DataFrame({'hold': holdCenter_ind, 
                       'reward': reward_ind, 
                       'pert': pert[reward_ind], 
                       'errorClamp': error_clamp[reward_ind], 
                       'blockType': block_type[reward_ind]})
    
    trialsStart = [np.where(df['blockType'] == i)[0][0]  for i in np.unique(block_type)]
    trialsEnd   = [np.where(df['blockType'] == i)[0][-1] for i in np.unique(block_type)]
    
    # if len(np.unique(block_type)) == 3:
    #     keys = ['BL', 'PE', 'PL']
    #     numB = 3
    # elif len(np.unique(block_type)) == 4:
    #     keys = ['BL', 'PE', 'PL', 'WO']
    #     numB = 4
    # else:
    #     print("ERROR IN NUMBER OF BLOCKS")
    keys = ['BL','PE']
    numB=1
        
        
    #First 1/3 and last 1/3
    # trialsEnd[1]   =  int(trialsStart[1] + (trialsEnd[2] - trialsStart[1])/3)
    # trialsStart[2] =  int(trialsEnd[2] -   (trialsEnd[2] - trialsStart[1])/3)
    
    decoder_state = hdf.root.task[:]['decoder_state']
    
    cursor = hdf.root.task[:]['cursor']
    x = cursor[:,0]
    y = cursor[:,2]
    
    cursorx = x
    cursory = y
    
    num_neurons = np.shape(hdf.root.task[:]['spike_counts'])[1]
    spikes = hdf.root.task[:]['spike_counts'] # (inds, neurons, 1)
    
    
    '###################################'
    ''' Determining Target Locations '''
    '###################################'
    
    dfTarget = pd.DataFrame(hdf.root.task[:]['target'])
    
    dfTarget['degrees'] = np.arctan2(dfTarget[2], dfTarget[0]) * (180/np.pi) 
    dfTarget['degrees'] = list(map(lambda d:d+360 if d < 0 else d, dfTarget['degrees']))
    
    #Trim to just the reward indices
    dfTargetLoc = pd.DataFrame(columns=['degrees', 'holdCenter_ind', 'reward_ind'])
    dfTargetLoc['degrees'] = dfTarget['degrees'][reward_ind]
    dfTargetLoc['holdCenter_ind'] = holdCenter_ind
    dfTargetLoc['reward_ind'] = reward_ind
    dfTargetLoc = dfTargetLoc.reset_index(drop=True)
    
    targetDeg = np.unique(dfTargetLoc['degrees'])
    
    df['target'] = dfTarget['degrees'][df['reward']].tolist()
    
    
    '############################'
    ''' Decoder update inds based on cursor values and checked against how many times KG was stored. '''
    '############################'
    
    lenHDF = len(hdf.root.task)
    lenKG  = len(KG_stored)
    
    posX = decoder_state[:,0,0]
    posY = decoder_state[:,2,0]
    
    velX = decoder_state[:,3,0]
    velY = decoder_state[:,5,0]
    
    first_update = np.min([np.where(posX > 0)[0][0], np.where(posY > 0)[0][0], np.where(velX > 0)[0][0], np.where(velY > 0)[0][0]])
    
    
    count = 0
    update_inds = []
    for i in range(lenHDF):
        if count < 5:
            update_inds.append(0)
            count+=1
        elif count == 5:
            update_inds.append(1)
            count = 0
    
    
    '############################'
    ''' Cursor and Time Plots '''
    '############################'
    
    dictTrials = {} #Without error clamp trials
    dictTimes  = {}
    
    dColors = dict(zip(keys,c))
    dTitles = dict(zip(keys,titles))
    dStart  = dict(zip(keys,trialsStart))
    dEnd    = dict(zip(keys,trialsEnd))
    
    
    for k in keys:
        dictTrials[k], dictTimes[k] = basicCursorTimes(df, x, y, dStart[k], dEnd[k])


    '################################################' 
    '################################################' 
    '################################################' 
    '################################################' 
    '################################################' 
    
    
    '################################################'    
    ''' Individual Unit Tuning Curves'''
    '################################################' 
    
    # Tuning curves based on every 100ms bin (#spikes) in trial.
    lenFA = [len(spikeCount(dictTrials[k], dfTargetLoc, spikes, 0, x, y, update_inds)) for k in keys]
    dlenFA = dict(zip(keys, lenFA))
    
    dSC   = {} #dict(zip(keys, np.zeros((len(lenFA), num_neurons)))) #dictionary of spike counts for each unit
    dTC   = {} #dict(zip(keys, (np.zeros((360,num_neurons)))))        #dictionary of tuning curves for each unit
    dPDMD = {} #dict(zip(k, np.zeros((num_neurons,2))))              #dictionary of preferred direction and modulation depth determined by tuning curve for each target
    dfSC  = {}
    nb = 8
    
    tuningPlot  = False
    
    sortbyPD = []
    sortbyTC = []
    
    
    for k in keys:
        dfSC[k]  = {}
        dTC[k]   = np.zeros((360, num_neurons))
        dPDMD[k] = np.zeros((num_neurons, 2))
        dSC[k]   = np.zeros((dlenFA[k], num_neurons)) 
    
    
        for neuron in range(num_neurons):
        
            dfSC[k][neuron] = spikeCount(dictTrials[k], dfTargetLoc, spikes, neuron, x, y, update_inds)
            

            dTC[k][:,neuron], dPDMD[k][neuron,0], dPDMD[k][neuron,1] =  \
                tuningCurve(dfSC[k][neuron], tuningPlot, units[neuron], dTitles[k], dColors[k])
            
            dSC[k][:,neuron] = dfSC[k][neuron]['SC']
            
            if (k == 'BL'):
                sortbyPD.append(dPDMD[k][neuron,0])
                sortbyTC.append(dTC[k][0,neuron])
                
                
    #os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Data\Airport')
    return(units, cursorx, cursory, spikes, dictTrials, dfSC, dTC, dPDMD, dfTargetLoc, update_inds, dictTimes)

def vectorPDChange(dPDMD, k1, k2, i):
    '''
    -Function to calculate the "absolute" change in preferred direction (PD; in degrees)
    -Only works for BL vs PE changes

    Parameters
    ----------
        dPDMD : dictionary containing all 
        k1    : key of first block (e.g. 'BL')
        k2    : key of second block (e.g. 'PE')
        i     : neuron/unit number
        
    Returns
    -------
        degChange : angle between the two PD vecotrs

    '''
    
    radA = (dPDMD[k1][i,0]) * (np.pi/180)
    radB = (dPDMD[k2][i,0]) * (np.pi/180)
    
    a = np.array( (np.cos(radA),np.sin(radA)) )
    b = np.array( (np.cos(radB),np.sin(radB)) )
    
    degChange = np.arccos(np.matmul(a,b)) * (180/np.pi)
    
    return(degChange)