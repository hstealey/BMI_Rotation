
'''
############################################################################################################################
     MULTIPLE rotation session (BMI Task: BMICursorRotErrorClamp in error_clamp_tasks.py)
        NOTE: This script EXCLUDES "error clamp" trials in analysis.
###########################################################################################################################    
    

Files Required to Run Script:

    Assumes that data is saved in the following heirarchy:
            Subject (e.g., Airport, Brazos)
                Degree Folder (e.g., 50, 90, neg50, neg90)
                    Folder for each session (e.g., 20220315, 20220316, 20220318)
                        3 Files/Folder ONLY: HDF, .pkl, .pkl


Analyses:



***Search '#!' for lines of code that need to be altered.***




@author: hanna
created on Sun Jan 23 12:18:06 2022
last updated: March 30, 2022
'''


#%%
import os
import pickle
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.sans-serif': 'Arial'})


#%%

subject = 'braz'
path = r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Data'

os.chdir(path)

'Open VARIANCE .pkl file'
var_file = glob(subject+'_KY_var.pkl')
open_file = open(var_file[-1], "rb")
deg_list, date_list, num_neurons, dFA, TP, x_pos, y_pos, KYx, KYy, TS, TP = pickle.load(open_file)
open_file.close()

'Open PATH LENGTH .pkl file'
pl_file = glob(subject+'_cursorPathLength.pkl')
open_file = open(pl_file[-1], "rb")
deg_list, date_list, trial_num, dist = pickle.load(open_file)
open_file.close()

#%%
#TS, TP:  [date][block] --> [0-7] (8, #neurons)



'''Shared Variance Over Sessions & Between Blocks'''
fsBL = np.zeros((8,len(date_list)))
fsPE = np.zeros((8,len(date_list)))

for d in range(8):
    fsBL[d,:] = [np.sum(TS[date]['BL'][d]) / np.sum(TS[date]['BL'][d] + TP[date]['BL'][d]) for date in date_list] 
    fsPE[d,:] = [np.sum(TS[date]['PE'][d]) / np.sum(TS[date]['PE'][d] + TP[date]['PE'][d]) for date in date_list]     

plt.plot(np.mean(fsBL, axis=0), color='k', label='Baseline')
plt.plot(np.mean(fsPE, axis=0), color='r', label='Rotation')
plt.ylim([0,1])

plt.title('Shared Variance')
plt.xlabel('Session Number')
plt.ylabel('Fraction of Total Shared Variance')
plt.legend(title='Block', loc='lower left')
plt.show()


#%%
'''Shared Variance v Distance/Behavior (within a session?)'''

#dist[date]['BL'][angle]
targetDeg = np.arange(0,360,45)



for d in range(1,2):
    plt.plot(dist[date]['BL'][targetDeg[d]])
    plt.plot(dist[date]['PE'][targetDeg[d]])



#%%
'''Shared Variance Histograms Between Blocks'''













