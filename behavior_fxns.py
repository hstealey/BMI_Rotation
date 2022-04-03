# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:54:00 2021

@author: hanna
"""

import numpy as np
import matplotlib.pyplot as plt

def basicCursorTimesPLOT(ax, title, color, df, x, y, trialStart, trialEnd):
    
    nonEC_trials = []
    times = []
    
    for i in range(trialStart, trialEnd):
        if df['errorClamp'][i] == 0:
            s = df['hold'][i]
            e = df['reward'][i]
            ax.plot(x[s:e], y[s:e], c=color)
            nonEC_trials.append(i)
            times.append( (e-s)/ 60)
            
    ax.set_title(title)

    return(nonEC_trials, times)


def basicCursorTimes(df, x, y, trialStart, trialEnd):
    
    nonEC_trials = []
    times = []
    
    for i in range(trialStart, trialEnd):
        if df['errorClamp'][i] == 0:
            s = df['hold'][i]
            e = df['reward'][i]
            nonEC_trials.append(i)
            times.append( (e-s)/ 60)
            
    return(nonEC_trials, times)


def plotTimes(keys, dictTrials, dictTimes, c, titles, TS, TE, deg, filename):
    mT = dict(zip(keys, [np.mean(dictTimes[k]) for k in keys]))
    sT = dict(zip(keys, [np.std(dictTimes[k]) for k in keys]))
    
    plt.figure(figsize=(10,5))
    plt.ylabel('Trial Time (s)', fontname='Arial', fontsize=16)
    plt.xlabel('Trial Number', fontname='Arial', fontsize=16)
    plt.title(r'Rotation (' + str(deg) + '$\degree$) - Session: ' + filename, fontname='Arial', fontsize=18)

    [plt.plot(dictTrials[k], dictTimes[k], c=c[k], label=titles[k]) for k in keys]
    [plt.fill_betweenx(y=[0,10], x1=[TS[k],TS[k]], x2=[TE[k],TE[k]], alpha=0.1, color=c[k]) for k in keys]
    [plt.text(TS[k], 0, str( np.round(mT[k],2) ) + ' +/- ' + str( np.round(sT[k],2)) + 's') for k in keys]
    
    plt.legend()
    plt.show()
    
    return()