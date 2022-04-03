# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:18:38 2021

@author: hanna
"""

import os
import glob
import tables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression
from scipy import stats
from factor_analyzer import FactorAnalyzer
import warnings
import pickle
import seaborn as sb

def performFA(x, num_neurons):
    
    '''
    Function to perform Factor Analysis and plot results
    
    inputs: 
        x: 2-D numpy array (samples (e.g. z-scored spike count for each 0.1ms update) x num_neurons)
        num_neurons: number of neurons
        
    returns:
        fa.loadings_: loadings or weights for each neurons and each factor
        ev: eigenvalue; >1 means that the factor explains the variance of more than 1 variable/neuron
    
    '''
    
    #Factor Analysis and Results
    fa = FactorAnalyzer(n_factors=num_neurons-1, rotation=None, method='ml')
    fa.fit(x)
    ev, v = fa.get_eigenvalues()
    variance, proportionalVariance, cumulativeVariance = fa.get_factor_variance()
    
    # #Cumulative Variance Plot
    print(cumulativeVariance[-1])
    plt.plot(cumulativeVariance, c='cornflowerblue', lw=3)
    plt.title('Cumulative Explain Variance \nTrained on Baseline Activity')
    plt.xlabel('Factor')
    plt.ylabel('Fraction of Total Variance')
    plt.xticks(ticks=np.arange(num_neurons), rotation=90)
    plt.show()
    
    #Eigenvalue Plot
    plt.plot(ev, c ='purple', lw=3)
    plt.title('Eigenvalues')
    plt.xlabel('Factor')
    plt.ylabel('Eigenvalue')
    plt.axhline(1, ls='--', c='k')
    plt.xticks(ticks=np.arange(num_neurons), rotation=90)
    plt.show()
    
    plt.plot(proportionalVariance)

    
    return(fa.loadings_, cumulativeVariance, proportionalVariance)


def spikeCount(inds, dfTargets, spikes, neuron, X, Y, update_inds):
 
    'Note: previously firingRate'
    'Note: changed from average over trial to each decoder updates (6 bins/100ms)'
    'Note (11/9/2021): change x and y from decoded velocity to cursor position'
    'Note (11/16/2021): change x and y back to decoded velocity '
    'Note (11/24/2021): change x and y back to cursor position'
    'Note (11/26/2021): changed updating scheme from difference in X or Y to update_inds'
   
    SC = []
    posX = []
    posY = []
    trial_num = []
    import numpy as np
    for j in inds:
        s = dfTargets['holdCenter_ind'][j]
        e = dfTargets['reward_ind'][j]
        
        for i in range(s,e):
            if update_inds[i] == 1:
                SC.append(np.sum(spikes[i-5:i+1, neuron, 0]))
                posX.append(X[i])
                posY.append(Y[i])
                trial_num.append(j)

    
    df = pd.DataFrame({'SC':SC, 'x':posX, 'y':posY, 'trial_num':trial_num}) 
    
    return(df)


def tuningCurve(df, plotTC, unit, block, c):
    '''Tuning Curve using Linear Regression (Ganguly & Carmena 2009)
    
    'Note (11/16/2021): Changed from using spike count to firing rate.'
    'Note (11/17/2021): Added MODULATION DEPTH (peak to peak ampitude of tuning curve in Hz)'
    'Note (11/30/2021): Changed to fiting curves based on average for each bin'
    

    inputs
        df: Pandas dataframe that contains the spike counts, velX, and velY for each time step (0.1s); output of def spikeCount()
        plotTC: boolean value indicating if the tuning curve should be plotted
        unit: neural index
        block: two-letter string indicating portion of task (e.g., baseline:'BL', perturbation-early:'PE')
        c: color
        
    returns
        tc: estimated tuning curve
        prefDir: float value for preferred direction of units
        modDep: float value for modulation depth (Hz)
        
    '''
    num_bins = 8
    
    #Bin and then fit linear regression tuning curve (Ganguly & Carmena2009).
    df['theta'] = np.arctan2(df['y'], df['x']) #Double check order of arguments
    df['theta'] = list(map(lambda r:r+(2*np.pi) if r < 0 else r, df['theta']))
    bins = np.arange(0,361*(np.pi/180), (360/num_bins)*(np.pi/180))
    labels = bins[:-1]
    df['bin'] = pd.cut(df['theta'], bins=bins, labels=labels)
    df['cos_theta'] = [np.cos(df['bin'][i]) for i in range(len(df))]
    df['sin_theta'] = [np.sin(df['bin'][i]) for i in range(len(df))]
    
    #Drop rows with NAN values. Likely caused by division by 0 in theta calculation.
    df = df.dropna()
    
    #Calculate and plot mean + SEM for each bin.
    m = [np.mean(df.loc[df['bin']==i, 'SC']/0.100) for i in bins[:-1]]
    s = [stats.sem(df.loc[df['bin']==i, 'SC']/0.100) for i in bins[:-1]]

    
    cosMean = np.cos(bins[:-1])
    sinMean = np.sin(bins[:-1])
    
    dfM = pd.DataFrame({'cos_theta': cosMean, 'sin_theta': sinMean, 'mFR': m})
    
    xy = dfM[['cos_theta', 'sin_theta']]
    FR = dfM['mFR']
    reg = LinearRegression().fit(xy, FR)
        
    B0 = reg.intercept_
    Bx = reg.coef_[0]
    By = reg.coef_[1]
    
    #Interpolate points for full range (0 - 2pi =  0 - 360 degrees).
    theta_tune = np.arange(0,360*(np.pi/180),1*(np.pi/180))
    x_tune = np.cos(theta_tune)
    y_tune = np.sin(theta_tune)

    tc = [Bx*x_tune[i] + By*y_tune[i] + B0 for i in range(len(theta_tune))]
    estMean = [(Bx*np.cos(i) + By*np.sin(i) + B0) for i in bins[:-1]]
    
    fix = lambda d:d+360 if d < 0 else d
    prefDir = fix(np.arctan2(By,Bx)*(180/np.pi))
    modDep = np.max(tc) - np.min(tc)
    
    if plotTC == True:
        
        plt.plot(bins[:-1], m, c=c, label='Mean Spike Rate +/- SEM')
        plt.fill_between(x=bins[:-1], y1=np.subtract(m,s), y2=np.add(m,s), alpha=0.2, color=c)
        
        #Plot tuning curve estimate based on linear regression model.
        plt.plot(theta_tune, tc, color='k', ls='--', label='Fitted Curve')
        plt.xticks(ticks=labels, labels=np.arange(0,360,(360/num_bins)), rotation=90)
        plt.title(block+' Tuning Curve for Unit '+unit)
        plt.ylabel('Average Spike Rate')
        plt.xlabel('Degrees (left edge of bin)')
        plt.legend()
        plt.show()
    
        

    return(tc, prefDir, modDep) 


def tuningCurveFA(PCsMult, df, plotTC, factor_num):
    
    '''Tuning Curve using Linear Regression - on FA-Reduced Data'''
    
    df['FA'] = PCsMult
    
    #Drop rows with NAN values.
    df = df.dropna()
    
    xy = df[['cos_theta', 'sin_theta']]
    FR = df['FA']
    
    reg = LinearRegression().fit(xy, FR)

    B0 = reg.intercept_
    Bx = reg.coef_[0]
    By = reg.coef_[1]            
    
    num_bins = 8
    bins = np.arange(0,361*(np.pi/180), (360/num_bins)*(np.pi/180))
    
    #Interpolate points
    theta_tune = np.arange(0,360*(np.pi/180),1*(np.pi/180))
    x_tune = np.cos(theta_tune)
    y_tune = np.sin(theta_tune)
    
    tc = [Bx*x_tune[i] + By*y_tune[i] + B0 for i in range(len(theta_tune))]
    
    
    modDep = np.max(tc) - np.min(tc)
    
    fix = lambda d:d+360 if d < 0 else d
    prefDir = fix(np.arctan2(By,Bx)*(180/np.pi))
    

    
    if plotTC == True:
        
        #Calculate and plot mean + SEM for each bin.
        m = []
        s = []
        for i in bins:
            temp = df.loc[df['bin']==i, 'FA']
            m.append(np.mean(temp))
            s.append(stats.sem(temp))
            
        plt.plot(bins, m, c='b', label='Mean +/- SEM')
        plt.fill_between(x=bins, y1=np.subtract(m,s), y2=np.add(m,s), alpha=0.2, color='b')
        
        #Plot tuning curve estimate based on linear regression model.
        plt.plot(theta_tune, tc, color='k', ls='--', label='Fitted Curve')
        plt.xticks(ticks=bins, labels=np.arange(0,361,(360/num_bins)).astype(int), rotation=90)
        plt.title('Tuning Curve for Factor ' + str(factor_num+1))
        plt.ylabel('Weighted+Summed Spike Counts?')
        plt.xlabel('Degrees (left edge of bin)')
        plt.legend()
        plt.show()
    

    return(stats.zscore(tc))#, prefDir, modDep)

def targetDirectionFR(targetDeg, dfTargetLoc, spikes, n, trials):
    
    FR = []
    targs = []
    trialInds = []
    
    for i in targetDeg:
        block = dfTargetLoc.loc[trials]
        tr = block.loc[block['degrees'] == i]
        
        for t in tr.index:
            s = dfTargetLoc['holdCenter_ind'][t]
            e = dfTargetLoc['reward_ind'][t]
            time = (e-s)/60
            FR.append(np.sum(spikes[s:e,n,0], axis=0)/time) #firing rate for a trial
            targs.append(i*(np.pi/180))
            trialInds.append(t)

    dfFR = pd.DataFrame({'trialInds': trialInds, 'FR': FR, 'targetRad': targs})
    PD, MD = tuningCurveTrialAverage(dfFR)
    
    return(PD, MD)

# def targetDirectionTwoBins(targetDeg, dfTargetLoc, spikes, n, trials, update_inds,num_bins, X, Y):
    
#     SC = []
#     trialInds = []
#     posX = []
#     posY = []
    
#     for i in targetDeg:
#         block = dfTargetLoc.loc[trials]
#         tr = block.loc[block['degrees'] == i]
        
#         for t in tr.index:
#             s = dfTargetLoc['holdCenter_ind'][t]
#             count = 0 
#             while count < 2:
#                 if update_inds[s] == 1:
#                     SC.append(np.sum(spikes[s-5:s+1, n, 0]))
#                     trialInds.append(t)
#                     posX.append(X[s])
#                     posY.append(Y[s])
#                     count+=1
#                 else: 
#                     s+=1
#                     count=0
                    
    
#     df = pd.DataFrame({'trialInds': trialInds, 'SC': SC, 'x':posX, 'y':posY})
        
#     PD, MD = tuningCurveTwoBins(df, num_bins)
    
#     return(PD, MD)

# def tuningCurveTwoBins(df, num_bins):
#     #Bin and then fit linear regression tuning curve (Ganguly & Carmena2009).
#     df['theta'] = np.arctan2(df['y'], df['x']) #Double check order of arguments
#     df['theta'] = list(map(lambda r:r+(2*np.pi) if r < 0 else r, df['theta']))
#     bins = np.arange(0,361*(np.pi/180), (360/num_bins)*(np.pi/180))
#     labels = bins[:-1]
#     df['bin'] = pd.cut(df['theta'], bins=bins, labels=labels)
#     df['cos_theta'] = [np.cos(df['bin'][i]) for i in range(len(df))]
#     df['sin_theta'] = [np.sin(df['bin'][i]) for i in range(len(df))]
    
#     #Drop rows with NAN values. Likely caused by division by 0 in theta calculation.
#     df = df.dropna()
    
#     #Perform linear regression. 
#     xy = df[['cos_theta', 'sin_theta']]
#     FR = df['SC']/0.100
#     reg = LinearRegression().fit(xy, FR)
    
#     B0 = reg.intercept_
#     Bx = reg.coef_[0]
#     By = reg.coef_[1]
              
#     #Interpolate points for full range (0 - 2pi; 0 - 360 degrees).
#     theta_tune = np.arange(0,360*(np.pi/180),1*(np.pi/180))
#     x_tune = np.cos(theta_tune)
#     y_tune = np.sin(theta_tune)
    
#     tc = [Bx*x_tune[i] + By*y_tune[i] + B0 for i in range(len(theta_tune))]
    
#     fix = lambda d:d+360 if d < 0 else d
#     prefDir = fix(np.arctan2(By,Bx)*(180/np.pi))
    
#     modDep = np.max(tc) - np.min(tc)
    
#     return(prefDir, modDep)


# def tuningCurveTrialAverage(dfFR):
#     #Perform linear regression. 
#     dfFR['cos_theta'] = np.cos(dfFR['targetRad'])
#     dfFR['sin_theta'] = np.sin(dfFR['targetRad'])
#     xy = dfFR[['cos_theta', 'sin_theta']]
#     FR = dfFR['FR']
#     reg = LinearRegression().fit(xy, FR)
    
#     B0 = reg.intercept_
#     Bx = reg.coef_[0]
#     By = reg.coef_[1]

#     #Interpolate points for full range (0 - 2pi; 0 - 360 degrees).
#     theta_tune = np.arange(0,360*(np.pi/180),1*(np.pi/180))
#     x_tune = np.cos(theta_tune)
#     y_tune = np.sin(theta_tune)
    
#     tc = [Bx*x_tune[i] + By*y_tune[i] + B0 for i in range(len(theta_tune))]
    
#     fix = lambda d:d+360 if d < 0 else d
#     prefDir = fix(np.arctan2(By,Bx)*(180/np.pi))
    
#     modDep = np.max(tc) - np.min(tc)
    
#     return(prefDir, modDep)


#%% DON'T USE! Method 2: Tuning Curves Based on first two updates (2x100ms)

# PD_MD2 = {'BL':np.zeros((num_neurons,2)),
#          'PE':np.zeros((num_neurons,2)),
#          'PL':np.zeros((num_neurons,2)),
#          'WO':np.zeros((num_neurons,2))}

# for n in range(num_neurons):
#     for k in keys:
#         PD_MD2[k][n,0], PD_MD2[k][n,1] = targetDirectionTwoBins(targetDeg, dfTargetLoc, spikes, n, dictTrials[k], update_inds, num_bins, x, y)



#%% DON'T USE! Method 3: Tuning Curves Based on TARGET DIRECTION and AVERAGE FIRING RATE 

# PD_MD3 = {'BL':np.zeros((num_neurons,2)),
#          'PE':np.zeros((num_neurons,2)),
#          'PL':np.zeros((num_neurons,2)),
#          'WO':np.zeros((num_neurons,2))}

    
# for n in range(num_neurons):
#     for k in keys:
#         PD_MD3[k][n,0], PD_MD3[k][n,1] = targetDirectionFR(targetDeg, dfTargetLoc, spikes, n, dictTrials[k])
    
    
