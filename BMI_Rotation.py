#For use in analyzing BMI rotation data from 11/6/2021 forward.


'''
############################################################################################################################
    Main script for basic analysis of a SINGLE rotation session (BMI Task: BMICursorRotErrorClamp in error_clamp_tasks.py)
        NOTE: This script EXCLUDES "error clamp" trials in analysis.
###########################################################################################################################    
    

Files Required to Run Script:
    decoder file (.pkl)
    Kalman gain file (...KG_VRKF.pkl)
    behavior + binned spikes file (.hdf)
    
    *NOTE: The way the code is currently written assumes only one of each file in the current working directory.


Analyses:
    Trial times and plot
    Cursor trajectories plot

    Tuning Curves
        individual units
        factors (from factor analysis)
        
    Amount of Learning (still somewhat under construction)


***Search '#!' for lines of code that need to be altered.***
    Directory paths
        bmi_python
        custom functions
        data
    Adjustable parameters
        boolean: to show individual unit tuning curves (or not)
        number of factors to display in FA heatmap


@author: hanna
created on Sun Nov  7 13:40:10 2021
last updated: March 30, 2022
'''



#%%


import os
import glob
import tables
import pickle
import numpy as np
import pandas as pd
import seaborn as sb
from   scipy import stats
import matplotlib.pyplot as plt
from   numpy.linalg import norm
from   collections import Counter
from   numpy import concatenate as ct 

import warnings
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.sans-serif': 'Arial'})

''' #! Change directory to BMI PYTHON location.'''
os.chdir(r'C:\Users\hanna\OneDrive\Documents\bmi_python-bmi_developed')
import riglib

''' #! Change directory to CUSTOM FUNCTIONS location.'''
os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation')
from tuningCurve_fxns import performFA, spikeCount, tuningCurve, tuningCurveFA
from behavior_fxns    import plotTimes, basicCursorTimesPLOT


'''#! Change directory to where data is stored.'''
os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Data\Brazos\neg50\20220328')  

hdf_files = glob.glob('*hdf')
hdf = tables.open_file(hdf_files[0])

decoder_files = glob.glob('*.pkl')
decoder_filename = decoder_files[0]
KG_picklename = decoder_files[1]

#Adjustable parameters
c = ['green', 'darkgoldenrod', 'blue', 'black']
titles = ['Baseline', 'Pert (Early)', 'Pert (Late)', 'Washout']

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


if len(np.unique(block_type)) == 2:
    keys = ['BL', 'PE']
    numB = 2
if len(np.unique(block_type)) == 3:
    keys = ['BL', 'PE', 'PL']
    numB = 3
elif len(np.unique(block_type)) == 4:
    keys = ['BL', 'PE', 'PL', 'WO']
    numB = 4
else:
    print("ERROR IN NUMBER OF BLOCKS")
    
    
decoder_state = hdf.root.task[:]['decoder_state']

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

dfTargetLoc = pd.DataFrame(columns=['degrees', 'holdCenter_ind', 'reward_ind'])
dfTargetLoc['degrees'] = dfTarget['degrees'][reward_ind]
dfTargetLoc['holdCenter_ind'] = holdCenter_ind
dfTargetLoc['reward_ind'] = reward_ind
dfTargetLoc = dfTargetLoc.reset_index(drop=True)

targetDeg = np.unique(dfTargetLoc['degrees'])
print('Unique target locations (degrees): ', targetDeg)

df['target'] = dfTarget['degrees'][df['reward']].tolist()


'############################'
''' Decoder update inds to sync binned spikes with task'''
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

fig, axes = plt.subplots(1, numB, figsize=(5*numB,4), sharex=True, sharey=True)
fig.suptitle(r'Cursor Trajectories - Rotation Perturbation (' + str(deg) + '$\degree$) - Session: ' + hdf_files[0][:15])

dictTrials = {} #Without error clamp trials
dictTimes  = {}

dColors = dict(zip(keys,c))
dTitles = dict(zip(keys,titles))
dStart  = dict(zip(keys,trialsStart))
dEnd    = dict(zip(keys,trialsEnd))
dAxes   = dict(zip(keys,axes))


'''Cursor Trajectories, Separated by Block'''
for k in keys:
    dictTrials[k], dictTimes[k] = basicCursorTimesPLOT(dAxes[k], dTitles[k], dColors[k], df, x, y, dStart[k], dEnd[k])
plt.show()


'''Individual Trial Times, Color-Coded by Block'''
plotTimes(keys, dictTrials, dictTimes, dColors, dTitles, dStart, dEnd, deg, hdf_files[0])
plt.show()



#%%
'################################################' 
'################################################' 
'################################################' 
'################################################' 
'################################################' 


'################################################'    
''' Individual Unit Tuning Curves'''
'################################################' 

# Tuning curves based on every 100ms bin in trial.

'''Determine how many update bins exist.'''
lenFA = [len(spikeCount(dictTrials[k], dfTargetLoc, spikes, 0, x, y, update_inds)) for k in keys] 
dlenFA = dict(zip(keys, lenFA))

dSC   = {} #Dictionary Spike Count                                   #dict(zip(keys, np.zeros((len(lenFA), num_neurons))))  
dTC   = {} #Dictionary Tuning Curve                                  #dict(zip(keys, (np.zeros((360,num_neurons)))))        
dPDMD = {} #Dictionary of Preferred Directions and Modulation Depths #dict(zip(k, np.zeros((num_neurons,2)))
dfSC  = {} #DataFrame of Spike Counts (used for factor analysis)     #dataframe()


tuningPlot  = False #! To display graph of tuning curve or not.

sortbyPD = []
sortbyTC = []

    
for k in keys:
    dfSC[k]  = {}
    dTC[k]   = np.zeros((360, num_neurons))
    dPDMD[k] = np.zeros((num_neurons, 2))
    dSC[k]   = np.zeros((dlenFA[k], num_neurons)) 

    for neuron in range(num_neurons):
        '''Align spike counts with each trial.'''
        dfSC[k][neuron] = spikeCount(dictTrials[k], dfTargetLoc, spikes, neuron, x, y, update_inds)
        
        '''Estimate the tuning curve, preferred direction, and modulation depth.'''
        dTC[k][:,neuron], dPDMD[k][neuron,0], dPDMD[k][neuron,1] =  \
            tuningCurve(dfSC[k][neuron], tuningPlot, units[neuron], dTitles[k], dColors[k])
        
        '''Save a copy of the '''
        dSC[k][:,neuron] = dfSC[k][neuron]['SC']
        
   
        if k == 'BL':
            '''Values used for sorting tuning plots.'''
            sortbyPD.append(dPDMD[k][neuron,0])
            sortbyTC.append(dTC[k][0,neuron])
                



#%%  
'################################################' 
'################################################' 
'################################################' 
'################################################' 
'################################################'

'################################################'    
''' Preferred Direction and Modulation Depth '''
'################################################' 


dfPD = pd.DataFrame({'neurons':units,'BL':dPDMD['BL'][:,0], 'PE':dPDMD['PE'][:,0], 'PL':dPDMD['PL'][:,0]}).sort_values(by=['BL'])
dfMD = pd.DataFrame({'neurons':units,'BL':dPDMD['BL'][:,1], 'PE':dPDMD['PE'][:,1], 'PL':dPDMD['PL'][:,1]}).sort_values(by=['BL'])

'################################################'    
''' PD Plotted by Block '''
'################################################' 

PDkeys = dfPD.keys()[1:]

modPE = [dfPD['PE'][i]+360 if (dfPD['BL'][i]-dfPD['PE'][i]) > 0 else dfPD['PE'][i] for i in dfPD.index.tolist()]
modPL = [dfPD['PL'][i]+360 if (dfPD['BL'][i]-dfPD['PL'][i]) > 0 else dfPD['PL'][i] for i in dfPD.index.tolist()]
size=175

plt.figure(figsize=((10,6)))
[plt.axvline(i, c='k', lw=0.5, alpha=0.2) for i in range(len(dfPD))]
plt.scatter(np.arange(len(dfPD)), dfPD['BL'], c=dColors['BL'],  label='BL', edgecolor='k', s=size)  
plt.scatter(np.arange(len(dfPD)), modPE,      c=dColors['PE'],  label='PE', edgecolor='k', s=size)
plt.scatter(np.arange(len(dfPD)), modPL,      c=dColors['PL'],  label='PL', edgecolor='k', s=size)
plt.title('Preferred Direction by Block - '+hdf.filename[:-4], fontname='Arial', fontsize=20)
plt.ylabel('Preferred Direction (' + (chr(176)) + ')', fontname='Arial', fontsize=16)
plt.xlabel('Unit', fontname='Arial', fontsize=16)
plt.xticks(ticks=np.arange(num_neurons), labels=dfPD['neurons'], rotation=90, fontname='Arial', fontsize=12)
plt.yticks(ticks=np.arange(0,np.max(modPL)+46,45), labels= ct((np.arange(0,361,45), np.arange(0, np.max(modPL)-360,45))), fontname='Arial', fontsize=12)
plt.legend(fontsize=18)
plt.show()


'################################################'    
''' HISTOGRAM: Changes in Preferred Direction '''
'################################################' 

def vectorPDChange(df, n, k1, k2):
    radA = (df[k1][n]) * (np.pi/180)
    radB = (df[k2][n]) * (np.pi/180)
    
    a = np.array( (np.cos(radA),np.sin(radA)) )
    b = np.array( (np.cos(radB),np.sin(radB)) )
    
    degChange = np.arccos(np.matmul(a,b)) * (180/np.pi)
    
    return(degChange)


dfPD = pd.DataFrame({'neurons':units,'BL':dPDMD['BL'][:,0], 'PE':dPDMD['PE'][:,0], 'PL':dPDMD['PL'][:,0]}).sort_values(by=['BL'])

PE_BL = [vectorPDChange(dfPD, n, 'PE', 'BL') for n in range(len(units))]

plt.figure(figsize=((5,3)))
plt.title('Change in Preferred Direction Due to Learning', fontweight='bold', fontsize=12.5)  
plt.ylabel('Number of Units in Bin', fontname='Arial')
plt.xlabel('Absolute Difference (' +  (chr(176)) + ')', fontname='Arial', fontsize=8, style='italic')
hist = plt.hist(PE_BL, color='blueviolet', edgecolor='k', alpha=0.8, bins=8)
plt.xticks(fontsize=8)
plt.scatter(np.mean(PE_BL), 12.5, c='blueviolet', edgecolor='k', s=75, alpha=1, label='Mean Change', zorder=3)
plt.scatter(50, 12.5, c='yellow', marker='*',s=150, edgecolor='k', zorder=3, label='Applied Rotation') 
plt.legend(fontsize=8)
plt.xlim([0,180])
plt.show()


    
'################################################'    
''' Modulation Depth Plotted by Block '''
'################################################' 

MDkeys = dfMD.keys()[1:]
plt.figure(figsize=((10,6)))
[plt.axvline(i, c='k', lw=0.5, alpha=0.2) for i in range(len(dfMD))]
[plt.scatter(np.arange(len(dfMD)), dfMD[k], c=dColors[k], label=k, edgecolor='k', s=size) for k in MDkeys]
plt.title('Modulation Depth by Block\n'+hdf.filename[:-4], fontname='Arial', fontsize=20)
plt.ylabel('Depth (Hz)', fontname='Arial', fontsize=16)
plt.xlabel('Unit', fontname='Arial', fontsize=16)
plt.xticks(ticks=np.arange(num_neurons), labels=dfMD['neurons'], rotation=90, fontname='Arial', fontsize=12)
plt.yticks(fontname='Arial', fontsize=12)
plt.legend(fontsize=18)
plt.show()


'################################################'    
''' HISTOGRAM: Changes in Modulation Depth  '''
'################################################' 

plt.figure(figsize=((5,3)))
dfMD_changes = pd.DataFrame({'units':units, 
                  'PE_BL':dPDMD['PE'][:,1] - dPDMD['BL'][:,1]}).sort_values(by=['PE_BL'])

plt.title('Modulation Depth Before and After Learning', fontweight='bold', fontsize=12.5)  
plt.ylabel('Number of Units in Bin', fontname='Arial')
plt.xlabel('Modulation Depth (Hz)', fontname='Arial', fontsize=10, style='italic')
b = plt.hist(dPDMD['BL'][:,1], color='k', bins=10, alpha=1)
plt.hist(dPDMD['PE'][:,1], color='blueviolet', edgecolor='k', bins=b[1], alpha=0.8)
plt.scatter(np.mean(np.abs(dPDMD['BL'][:,1])), 12, marker='d', c='k', s=60, alpha=1, edgecolor='k', label='Mean Before')#, s=70)
plt.scatter(np.mean(np.abs(dPDMD['PE'][:,1])), 12, marker='d', c='blueviolet', s=60, alpha=0.8, label='Mean After')
plt.legend(fontsize=7, ncol=1)
plt.show()



    
#%%
'################################################' 
'################################################' 
'################################################' 
'################################################' 
'################################################'

'################################################'    
''' Heatmaps for Tuning Curves'''
'################################################' 


def tcUnitHeatMap(tc, ax, pd_BL, first, title, units):
    
    temp = pd.DataFrame(tc)
    temp['units'] = units
    temp['value'] = pd_BL
    temp['first'] = first
    temp1 = temp.sort_values(by=['value', 'first'])
    unitsSorted = temp1['units']
    tcSorted = temp1.drop(columns=['first','value', 'units'])
    
    cbar = True
    
    cmap='PRGn'
        
    if ax == ax3:
        cmap = 'seismic'
    else:
        cmap='PRGn'
    
    sb.heatmap(tcSorted, ax=ax, cbar=cbar, cmap=cmap)
    ax.set_title(title, fontname='Arial', fontsize=20)
    
    ax.set_xlabel('Degrees', fontname='Arial', fontsize=16)
    ax.set_xticks(np.arange(0,361,45))
    ax.set_xticklabels(np.arange(0,361, 45).astype(int), rotation=45, fontsize=10)

    ax.set_yticks(np.arange(0,len(units)))
    ax.set_yticklabels(np.arange(1,len(units)+1), rotation=360)
    ax.set_ylabel('Unit Number', fontname='Arial', fontsize=16)
    
    return()


def tcHeatMapFA(tc, ax, title, count):
    
    temp = pd.DataFrame(tc)
    tcSorted = temp

    if (count == 1) or (count == 2):
        cbar=True
    else: 
        cbar=False
        
    if (count == 2):
        cmap = 'RdGy'#'seismic'
    else:
        cmap='PRGn'
        
    if count == 0:
        title = 'Baseline'
    elif count == 1:
        title = 'Perturbation'
    elif count == 2:
        title = 'Difference'
        
    sb.heatmap(tcSorted, ax=ax, cbar=cbar, cmap=cmap)
    ax.set_title(title, fontname='Arial', fontsize=24)
    ax.set_xlabel('Degrees', fontname='Arial', fontsize=18)
    ax.set_xticks(np.arange(0,361,45))
    ax.set_xticklabels(np.arange(0,361,45), rotation=45)

    
    if (count == 0):
        ax.set_ylabel('Factor Number', fontname='Arial', fontsize=30)
        ax.set_yticks(np.arange(0,10)+0.5)
        ax.set_yticklabels(np.arange(1,11), fontsize=18, rotation=0)
    else:
        ax.set_yticklabels([])
        
    return()


#%%

'################################################'    
''' Heatmap of Tuning Curves for Individual Units '''
'################################################' 

dTitles = {'BL': 'Baseline',
           'PE': 'Perturbation'}

fig, [ax1, ax2, ax3] = plt.subplots(1,3, sharey=False, figsize=((15,10)))
fig.suptitle('Changes in Normalized Tuning Curves', fontweight='bold', fontsize=24)# + hdf_files[session_num] + '\n 50' + u'\N{DEGREE SIGN} Rotation Perturbation', fontname='Arial', fontsize=22)
fig.subplots_adjust(top=0.9)
dAX = dict(zip(keys[:2], [ax1,ax2]))
[tcUnitHeatMap( stats.zscore(dTC[k].T, axis=1),  dAX[k], sortbyPD, sortbyTC, dTitles[k], units) for k in keys[:2]]
tcUnitHeatMap(dTC['PE'].T - dTC['BL'].T, ax3, sortbyPD, sortbyTC, 'PE Minus BL', units)

plt.show()




 #%%

'################################################'    
''' Heatmap of Factors '''
'################################################' 


num_factors = 10 #!  Number of factors to display in heatmap

w, cumulativeVariance, propVar =  performFA( stats.zscore(dSC['BL'], axis=0), num_neurons) 

pairsFAMult = []
pairsTCFA = []
for k in keys:
    pairsTCFA.append(  (k,np.zeros((num_factors,360)))   )  
    pairsFAMult.append(  (k, (np.matmul(dSC[k], w))) )
dictFAMult = dict(pairsFAMult)
dictTCFA = dict(pairsTCFA)



for factor in range(num_factors):
    for k in keys: 
        dictTCFA[k][factor,:] = tuningCurveFA(dictFAMult[k][:,factor], dfSC[k][0] , False, factor)


fig, [ax1, ax2, ax3] = plt.subplots(1,3, sharey=False, figsize=((16,10)))
axes = [ax1, ax2, ax3]
fig.suptitle('Changes in Normalized Factor Tuning Curves', fontweight='bold', fontsize=36)
i=0
for k in keys[:2]:
    tcHeatMapFA(dictTCFA[k][:num_factors,:],  axes[i], titles[i], i)
    i+=1

tcHeatMapFA(np.subtract(dictTCFA['PE'][:num_factors,:], dictTCFA['BL'][:num_factors,:]), axes[2], 'Pert. Minus Baseline', i)
plt.show()






#%%

''''Still under development'''

''''For more information: Neural constraints on learning by Sadtler et al., Nature 2014 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4393644/)'''

'################################################' 
'################################################' 
'################################################' 
'################################################' 
'################################################'

'################################################'    
''' Amount of Learning'''
'################################################' 

indTargPen = ind[np.where(hdf.root.task_msgs[:]['msg'] == b'timeout_penalty')[0]]
indHoldPen = ind[np.where(hdf.root.task_msgs[:]['msg'] == b'hold_penalty')[0]]

indEC_TP = np.where(error_clamp[indTargPen] == 1)[0]
indEC_HP = np.where(error_clamp[indHoldPen] == 1)[0]

iHP = np.delete(indHoldPen, indEC_HP)
iTP = np.delete(indTargPen, indEC_TP)

''''############################################################'''
def AoL(k, df, dictTimes, iHP, iTP):
    
    if k=='BL':
        start = np.arange(0,300,50)
        end   = np.arange(50,350,50)
    else:
        start = np.arange(0,350,50)
        end   = np.arange(50,400,50)
    
    mT = []
    pF = []
    
    if k == 'BL':
        mod      = len(dictTimes['BL'])//50
        modStart = len(dictTimes['BL']) - (mod*50)
        
        dfBlock = df.loc[dictTrials[k][modStart:]].reset_index(drop=True)
    else:
        dfBlock = df.loc[dictTrials[k][:]].reset_index(drop=True)
    
    
    for s,e in list(zip(start,end)):
        sub = dfBlock[s:e]
        subMin = np.min(sub['hold'])
        subMax = np.max(sub['hold'])
        
        count=0
        for i in iHP:
            if (i > subMin) & (i < subMax):
                count+=1
        for i in iTP:
            if (i > subMin) & (i < subMax):
                count+=1
                
        mT.append(np.mean(dictTimes[k][s:e]))
        pF.append( ((50-count) / 50) * 100)
    
    return(mT, pF)


def zAoL(MT, ST, MP, SP, i, j):
    nT = (i-MT)/ST
    nP = (j-MP)/SP
    
    return(nP, nT)

''''############################################################'''
mT = {}
pF = {}

for k in keys[:2]:
    mT[k], pF[k] = AoL(k, df, dictTimes, iHP, iTP)

MT = np.mean(sum(mT.values(), []))
ST = np.mean(sum(mT.values(), []))

MP = np.mean(sum(pF.values(), []))
SP = np.mean(sum(pF.values(), []))


nP_BL, nT_BL   = zip(*[zAoL(MT,ST,MP,SP,i,j) for i,j in zip(mT['BL'],pF['BL'])])
pBL            =  [np.mean(nP_BL), np.mean(nT_BL)]
pPE            = list(zip([zAoL(MT,ST,MP,SP,i,j) for i,j in zip(mT['PE'],pF['PE'])]))

#minPerformance = [np.linalg.norm(pPE[i]) for i in range(7)]
#minInd =  np.where(minPerformance == np.min(minPerformance))[0][0]

Lsession = []
for i in range(7):
    Lmax = np.subtract(pBL, pPE[0][0])  #initial performance impairment; isn't maximum, though????
    multiplier = Lmax/(np.linalg.norm(Lmax))
    Lraw = np.subtract(pPE[i][0], pPE[0][0]) #raw learning vector
    Lproj = (np.dot(Lraw, multiplier)) * multiplier #projected learning vectors
    Lbin = np.linalg.norm(Lproj) / np.linalg.norm(Lmax) #amount of learning (0 = no improvement, 1 = full improvement up to baseline block)
    Lsession.append(Lbin)
print(Lsession)

AoL = np.max(Lsession)



'################################################'    
''' Task Performance Plot'''
'################################################'


rP_BL = ct(([np.repeat(i,50) for i in ct((pF['BL'], pF['PE']))])) 

BL_PEtrials = np.concatenate((pd.Series(dictTimes['BL']).rolling(30).mean() , pd.Series(dictTimes['PE']).rolling(30).mean()  ))#pd.Series(ct(([dictTimes[k] for k in keys[:2]])))

fig, ax1 = plt.subplots()
ax1.set_xlabel('Trial Number')
ax1.set_ylabel('Success Rate (%)')
ax1.plot(np.arange(36, (6*50)+(7*50)+36, 1), rP_BL, c='k')
ax1.axvline(dictTrials['PE'][0], c='r', ls='--')
ax1.set_ylim([60,101])

ax2 = ax1.twinx()
ax2.set_ylabel('Acquistion Time (s)', color='g')
ax2.plot(np.arange(36,718+36), BL_PEtrials , c='g') #Fix the x-axis
ax2.tick_params(axis='y', labelcolor='g')
ax2.set_ylim([1.5,3])

plt.title('Task Performance - '+hdf.filename)
plt.show()


#%%
'################################################'    
''' Quantifying Amount of Learning Plot '''
'################################################'

maxAoLInd = np.where(Lsession == AoL)[0][0]
p1X = pPE[0][0][0]
p2X = pPE[maxAoLInd][0][0]

p1Y = pPE[0][0][1]
p2Y = pPE[maxAoLInd][0][1]

plt.title('Quantifying Amount of Learning - '+hdf.filename[:-4])
plt.xlabel('Success Rate (%, z-scored)')
plt.ylabel('Acquisition Time (s, z-scored)')

plt.scatter(pBL[0], pBL[1], c='k', edgecolor='k')
plt.plot([p1X, p2X], [p1Y, p2Y], c='r')
plt.plot([p1X, pBL[0]], [p1Y, pBL[1]], c='k', ls='--')
plt.scatter(p1X, p1Y, c='r', edgecolor='k', s=100)
plt.scatter(p2X, p2Y, c='r', marker='x', s=100)

plt.text(pBL[0]+0.001, pBL[1], 'Max. Learning Vector')
plt.text(p1X+0.001, p1Y, 'Initial (0)')
plt.text(p2X+0.001, p2Y, 'Max (3)')

#Perturbation Points
[plt.scatter(pPE[i][0][0], pPE[i][0][1], c='pink', edgecolor='k') for i in [1,2,4,5,6]]
[plt.text(pPE[i][0][0]+0.001, pPE[i][0][1], i, c='k') for i in [1,2,4,5,6]]

#Baseline Points - Currently, average is used for AoL statistic.
[plt.scatter(nP_BL[i], nT_BL[i], color='gray', edgecolor='k') for i in range(6)]
[plt.text(nP_BL[i]+0.001, nT_BL[i], i, c='k') for i in [0,1,2,3,4,5]]

# yLabs = [np.round((ST*i) + MT, 1) for i in np.arange(-0.2,0.3, 0.05)]
# xLabs = [int(np.round((SP*i) + MP)) for i in np.arange(-0.06,0.06, 0.01)]

# plt.xticks(ticks=np.arange(-0.06,0.06, 0.01), labels=xLabs)
# plt.yticks(ticks=np.arange(-0.2,0.3, 0.05), labels=yLabs)

plt.show()



    














