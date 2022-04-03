# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:48:08 2022

@author: hanna
"""


from glob import glob
import os
import pickle
import tables
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis


os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation')
from behavior_fxns import basicCursorTimes
from basic_fxns import trialsNStuff



def extractKG(date):
        os.chdir(date)
        
        pkl = glob('*.pkl')
        
        if 'KG' not in pkl[1]:
            print('INCCORECT PICKLE FILE: ', pkl[1])
        
        f = open(pkl[1], 'rb')
        KG_stored = []
        while True:
            try:
                KG_stored.append(pickle.load(f))
            except Exception:
                break
        f.close()
        
        #should really add a check to make sure these are in BL,PE...

        KG = pd.DataFrame({'xBL': np.array(KG_stored[5000][3][0,:]).reshape(-1,),
                                 'yBL': np.array(KG_stored[5000][5][0,:]).reshape(-1,),
                                 'xPE': np.array(KG_stored[20000][3][0,:]).reshape(-1,),
                                 'yPE': np.array(KG_stored[20000][5][0,:]).reshape(-1,)})

        os.chdir('..')
        return(KG)
    
    
    
    
    
def cursorPathLength(date):
    
        os.chdir(date)
    
        hdf = tables.open_file(glob('*.hdf')[0])
    
        '''Parse HDF file.'''
        msg = hdf.root.task_msgs[:]['msg'] #b'wait', b'premove', b'target', b'hold', b'targ_transition', b'target', b'hold', b'targ_transition', b'reward'
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
        
        cursor = hdf.root.task[:]['cursor']
        x = cursor[:,0]
        y = cursor[:,2]
        
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
        ''' Cursor and Trial Times '''
        '############################'
        dictTrials = {} #Without error clamp trials
        dictTimes  = {}
        
        keys = ['BL', 'PE']
        dStart  = dict(zip(keys,trialsStart))
        dEnd    = dict(zip(keys,trialsEnd))
        
        
        for k in keys:
            dictTrials[k], dictTimes[k] = basicCursorTimes( df, x, y, dStart[k], dEnd[k])
        
        
        'Total Distance Cursor Travels (i.e., path length)'
        
        dist = {}
        trial_num = {}
        
        for k in keys:
            trials = dictTrials[k]
            dist[k] = {}
            trial_num[k] = {}
            
            for d in targetDeg:
                dist[k][d] = []
                trial_num[k][d] = []
                block = dfTargetLoc.iloc[trials]
                tr = block.loc[block['degrees'] == d]
                start_inds = tr['holdCenter_ind']
                end_inds = tr['reward_ind']
                
                count = 0
                for i,j in zip(start_inds, end_inds):
        
                    x_ = [x[i]]
                    y_ = [y[i]]
                    
                    for q in range(i+1,j):
                        if (x[q] != x_[-1]) or (y[q] != y_[-1]):
                            x_.append(x[q])
                            y_.append(y[q])
                    
                    dx = np.diff(x_)
                    dy = np.diff(y_)
                    
                    dist[k][d].append(np.sum(np.sqrt(dx**2 + dy**2)))
                    trial_num[k][d].append(count)
                        
                    count+=1
                            
        os.chdir('..')
        return(dist, trial_num)
    
    

    
    
def KY(date, key, KGx, KGy):
    
    '''
    This function calculates the instanenous neural commands for x and y velocities.
    Additionally, it formats the spike counts for each unit across trials into one matrix (to be used for factor analysis).
        see: def varianceSharedPrivate
    

    inputs
        date: name of folder that contains the session data (format: YYYYMMDD)
        key: string - two-letter string that indicates the block (e.g., 'BL', 'PE')

        KGx: list - Kalman gain for each unit (x-velocity)
        KGy: list - Kalman gain for each unit (y-velocity)
            
    returns
        num_neurons: int - number of units used in session decoder
        dFA: dictionary - contains a matrix of spike counts from all trials and all neurons to be used in factor analysis
        TP: list of ints - timepoint/index of update within a trial
        x_pos: np.array of floats - cursor x-position (HDF)
        y_pos: np.array of floats - cursor y-position (HDF)
        KYx: neural command (x)
        KYy: neural command (y)
         
    '''
    
    units, _, _, spikes, dictTrials, dfSC, _, dPDMD, dfTargetLoc, _, dictTimes = trialsNStuff(date)
    num_neurons = len(units)

    dFA = {}
    TP  = {}
 
    x_pos = {}
    y_pos = {}

    trials = dictTrials[key]
 
    KYx = {}
    KYy = {}
   
    targetDeg = np.arange(0,360,45)
    for d in range(len(targetDeg)):
        degTrials = dfTargetLoc.loc[dfTargetLoc['degrees'] == targetDeg[d]].index
        nDEG = set(trials).intersection(degTrials)

        dFA[d] = np.zeros(( len(nDEG), num_neurons ))
        x_pos[d] = []
        y_pos[d] = []
        TP[d] = []
        
        KYx[d] = {}
        KYy[d] = {}
        
    
        for n in range(num_neurons):
            tSC = []
            KYx[d][n] = []
            KYy[d][n] = []
            
            trial=0
            for t in nDEG:
                
                'Determines the df indices for a trial.'
                inds = dfSC[key][n]['trial_num'].loc[dfSC[key][n]['trial_num'] == t].index
                
                'Spike counds for a trial.'
                SC_neuron_trial = [dfSC[key][n]['SC'][i] for i in inds]
                
                'Neural command for x and y velocities'
                KYx[d][n].append( np.multiply(SC_neuron_trial, KGx[n]))
                KYy[d][n].append( np.multiply(SC_neuron_trial, KGy[n]))
                trial+=1
                
                if n == 0:    
                    'x and y position'
                    x_pos[d].append([dfSC[key][n]['x'][i] for i in inds])
                    y_pos[d].append([dfSC[key][n]['y'][i] for i in inds])
                    
                    'Timepoint/index within a trial'
                    TP[d].append(np.arange(len(inds)))

                tSC.append(np.mean(SC_neuron_trial))
                
                
            dFA[d][:,n] = tSC 
                
                
    return(num_neurons, dFA, TP, x_pos, y_pos, KYx, KYy)
               
                
                
                
def varianceSharedPrivate(date, num_neurons, num_components, dFA, start=0):

    '''
    This function uses factor analysis to parse the total population variance into
    shared and private components. These calculations are performed separately 
    for each target direction and each block.
    
    
    inputs
        date: name of folder that contains the session data (format: YYYYMMDD)
        num_neurons: int - number of units used for BMI decoder
        num_components: int - TBD
        dFA: dictionary - contains a matrix of spike counts from all trials and all neurons to be used in factor analysis
        start: int - starting index of trials to analyze
            NOTE: for PE, previously used start=20
            
    returns
        TS: list of 8 floats - total shared variance as determined by factor analysis
        TP: list of 8 floats - total private variance as determined by factor analysis
    
    '''
    

    TS = []
    TP = []    

    div = num_components
    n_components= int(num_neurons/div)    
    
    targetDeg = np.arange(0,360,45)
    for d in range(len(targetDeg)):
        
            FA = FactorAnalysis(n_components=n_components,max_iter=10000)
            FA.fit(dFA[d][start:, :])
            shared_variance = FA.components_.T.dot(FA.components_)
            private_variance = np.diag(FA.noise_variance_)
            
            tot_shared  = [shared_variance[i,i] for i in range(num_neurons)]
            tot_private = [private_variance[i,i] for i in range(num_neurons)]
        
            TS.append( tot_shared )
            TP.append( tot_private )
        
    return(TS, TP)


    