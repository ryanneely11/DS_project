##ml_regress.py

##contains functions for running multiple linear regressions

import numpy as np
#import analyze_behavior as ab
import parse_timestamps as pt
import parse_ephys as pe
from sklearn import linear_model
from sklearn.feature_selection import f_regression

"""
This function runs a regression analysis on all of the
data from one session.
Inputs:
	-f_behavior: file path to behavior data (HDF5)
	-f_ephys: file path to ephys data (HDF5)
	-window: the time around behavioral events to look at spike
		data, in seconds, as a list like this: [2,1,1] 
		the first window is for secs before action, 
		the second is for time before outcomes/reward,
		the third is for time after reward
Returns:

"""
def session_regression(f_behavior,f_ephys,window=[2,1,1]):
	result = {} ##dictionary to return
	#get the behavioral data:
	behavior_ts = pt.sort_by_trial(f_behavior,save_data=False)
	block_names = behavior_ts.keys() ##names of the blocks
	behavior_data = None
	regressors = None
	##concatenate both regressors and behavior ts
	##at the same time to preserve order for the next step
	for n in block_names:
		block_data = behavior_ts[n]
		regr_data = get_regressors(block_data,n)
		if behavior_data != None:
			behavior_data = np.vstack((behavior_data,block_data))
		else:
			behavior_data = block_data
		if regressors != None:
			regressors = np.vstack((regressors,regr_data))
		else:
			regressors = regr_data
	##now work on the ephys data
	signals = pe.get_spike_data(f_ephys) ##the binary spike data
	intervals = ['pre-action','pre-reward','post-reward']##the three periods to look @
	triggers = ['action','outcome','outcome'] #ehhh... this will make sense later
	##generate the windows
	windows = ([window[0],0],[window[1],0],[0,window[2]])
	for n, inter in enumerate(intervals):
		##create a nested dictionary for this interval that will go in our output dict
		units_dict = {}
		##get the corresponding timestamp data to use
		trigger_type = triggers[n]
		##get the window for this interval
		win = windows[n]
		##run through each unit and calculate the regression
		for sig in signals.keys():
			data = signals[sig]
			##get the regressand data
			y = get_regressand(behavior_data,trigger_type,data,win)
			X = regressors
			##get the regression data for this unit in this interval
			coef,pvals = run_regression(y,X)
			##now save this data to the dictionary
			units_dict[sig] = (coef,pvals)
		##now save this data to the outermost dictionary
		result[inter] = units_dict
	return result


"""
a function to generate regressors (ie behavior values) 
from log data from one block. 
Regressors (6):
	-C(t): choice (upper (C=1) or lower (C=-1) lever)
	-R(t): outcome (rewarded (R=1) or unrewarded (R=0)
	-X(t): choice-reward outcome (C x R)
	-Qu(t): Action value of upper lever (0 or 0.85)
	-Ql(t): Action value of lower lever (0 or 0.85)
	-Qc(t): Action value of chosen lever (0 or 0.85)
Inputs: 
	-block_data: a numpy array of the format returned by
		pt.sort_block(). Basically one of the dictionary entries
		in a dictionary returned by sort_by_trial.
	-block_type: str, either "upper rewarded" or "lower rewarded."

Outputs:
	-X: array of regressors (n-samples x n-features)
"""
def get_regressors(block_data,block_type):
	##the input array has info about trial start, choice,
	##outcome, and reward value; but we need to parse it a little further
	##first figure out how many trials we are dealing with
	trials = block_data.shape[0]
	##setup the output array (n trials x 6 regressors)
	X = np.zeros((trials,6))
	##parse the data, one regressor at a time:
	if block_type == 'lower_rewarded':
		Ql = 0.85
		Qu = 0.0
	elif block_type == 'upper_rewarded':
		Qu = 0.85
		Ql = 0.0
	else:
		raise KeyError("unknown block type: "+key)
	##run through each trial in this block
	for i in range(block_data.shape[0]):
		##C(t)
		if block_data[i,1] < 0: ##case where lower lever is pressed
			X[i,0] = -1
		elif block_data[i,1] > 0: ##case where upper lever is pressed
			X[i,0] = 1
		else:
			raise ValueError("error parsing choice for trial "+str(i)+" in block "+key)
		#R(t)
		if block_data[i,2] < 0: ##case where unrewarded
			X[i,1] = 0
		elif block_data[i,2] > 0: ##case where rewarded
			X[i,1] = 1
		#X(t)
		X[i,2] = X[i,0]*X[i,1]
		#Qu(t)
		X[i,3] = Qu
		#Ql(t)
		X[i,4] = Ql
		#Qc(t)
		if X[i,0] < 0: #lower lever chosen
			X[i,5] = Ql
		elif X[i,0] > 0: ##upper lever chosen
			X[i,5] = Qu
		else:
			raise ValueError("Error calculating Qc(t) for trial "+str(i)+" in block "+key)
	return X


"""
This function returns the regressand for a block (which in our case is the mean
FRs in a given interval for a given neuron)
Inputs:
	-block_data: a numpy array of the format returned by
		pt.sort_block(). Basically one of the dictionary entries
		in a dictionary returned by sort_by_trial
	-trigger_type: str; either "action", or "outcome"
	-signal: the binary spike array 
	-window_size: list; the window around the timestamps to take data; ie [3,3]
Returns:
	-an n-trial array of the average spike rate in the desired window
"""
def get_regressand(block_data,trigger_type,signal,window=[3,3]):
	##a lookup table to go between str values and the correct index
	##in the block data
	trigger_lut = {
	"action":1,
	"outcome":2
	}
	trigger = trigger_lut[trigger_type]
	##get the appropriate timestamps
	timestamps = block_data[:,trigger]
	##get the windowed signal data
	data_windows,no_data = pe.data_windows(timestamps,signal,window)
	##figure out how many seconds are in the window so we can calculate the spike rate
	secs_per_window = float(window[0]+window[1])
	##get the spike count for each trial and divide by the window duration
	regressors = data_windows.sum(axis=1)/secs_per_window
	return regressors

"""
A function to run a multiple linear regression of the form:
S(t) = a0 + a1C(t) + a2R(t) + a3X(t) + a4Qu(t) + a5Ql(t) + a6Qc(t) + e(t); where
	-S(t) is the mean spike rate over some interval in trial t;
	-a0 thru a6 are the parameters to fit
	-C(t) is the choice of lever (upper(1) or lower(-1))
	-R(t) is the outcome (rewarded(1) or unrewarded(0))
	-X(t) is the interaction between choice and outcome (C*R)
	-Qu(t) is the value of the upper lever in the current block
	-Ql(t) is the value of the lower lever in the current block
	-Qc(t) is the value of the chosen action
	-e(t) is an error term
Inputs:
	-S: a vector of regressands, which is the mean spike rate over 
		an interval of interest for n trials (see get_regressands)
	-X: an n-trials by m-regressors array (see get_regresssors)
Returns:
	-coeff: fitted coefficient values
	-pvals: significance of fitted coeffiecient values based on t-test
"""
def run_regression(S,X):
	##initialize the regression
	regr = linear_model.LinearRegression()
	##fit the model
	regr.fit(X,S)
	##get the coefficients
	coeff = regr.coef_
	##now test for significance
	F, pvals = f_regression(X,S)
	return coeff, pvals


