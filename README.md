#Usage Information


This code is used to analyze brain-machine interface (BMI) data from a visusomotor rotation task.  

For task code, see https://github.com/santacruzlab  (branch: bmi_ssd) bmi_python>built_in_tasks>error_clamp_tasks.py: class BMICursorVisRotErrorClamp.


**BMI_Rotation.py**
	This script is used to analyze a single session from a rotation task.

	Analyses:
		Behavioral:
			Individual trial times
			Cursor trajectory plots
			Amount of Learning (see Sadtler et al., Nature 2014 for more information)
				*Note: Code is "under construction".
		Neural:
			Tuning curve estimation for individual units
				preferred direction (PD)
				modulation depth (MD)
			Tuning curve estimation for factors of units
				Factors determined by factor analysis (FA)
		

	Prerequisites:
		basic_fxns.py
		behavior_fxns.py
		tuningCurve_fxns.py


**generatePickles.py**

	Prerequisites:
		generatePickles_fxns.py


**sharedVariance.py**





#Updates
04.07.2022: Updated README