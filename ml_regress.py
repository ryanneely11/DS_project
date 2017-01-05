##ml_regress.py

##contains functions for running multiple linear regressions

import numpy as np
#import analyze_behavior as ab
import parse_timestamps as pt
import parse_ephys as pe
from sklearn import linear_model
from sklearn.feature_selection import f_regression
from scipy.stats import binom_test
import h5py
import multiprocessing as mp
import file_lists 


"""
A function to run regression analyses on a giant list of files
Inputs:
	-epoch_durations: a list of the desired durations (in sec) of each behavioral epoch. 
		the order is: choice (pre-action), peri-action (centered around the action ts),
		delay (time before the rewarde ts), and reward (time after the rewarde ts).
	-win_size = the size of each window  (in sec)
	-win_step = the time in s over which to slide the window over each epoch interval. 
		win_size = win_step means windows do not overlap. 
	-smooth: whether or not to use gaussian smoothing. Any value > 0 will smooth with
		a kernel of width smooth
Returns: 
	-coeffs: average coefficients for each regressor over the population
	-p_significant: an array where each data point is the fraction of units
		with significant correlation coefficients at a particular timepoint
	-epoch_idx: a dictionary reporting which indices in the window (time) dimenstion 
		correspond to which epochs
	TODO: add this feature [-sig_level: the level above which the num_significant values are significant
		(binomial test)] 
"""
def regress_everything(epoch_durations=[2,0.5,1,2],win_size=0.5,win_step=0.25,smooth=10):
	##data to return
	coeffs = None
	perc_sig = None
	num_total_units = 0
	##run session regression on all files in the lists
	for f_behavior,f_ephys in zip(file_lists.behavior_files,file_lists.ephys_files):
		c,ps,epoch_idx,num_units = regress_session_epochs(f_behavior,f_ephys,epoch_durations,
			win_size,win_step,smooth,proportions=False)
		num_total_units += num_units
		try:
			coeffs = coeffs + c
			perc_sig = perc_sig + ps
		except TypeError:
			coeffs = c
			perc_sig = ps
	##divide by total number of units
	coeffs = coeffs/float(num_total_units)
	perc_sig = perc_sig/float(num_total_units)
	return coeffs,perc_sig,epoch_idx,num_total_units



"""
This function regresses spike data from all individual units in a session,
and then returns arrays where each data point is the fraction of units with
significant regression coefficients at a particular time point in an epoch.
Inputs:
	-f_behavior: the file path of the HDF5 file contianing the behavioral data
	-f_ephys: the file path of the HDF5 file containing the ephys data
	-epoch_durations: a list of the desired durations (in sec) of each behavioral epoch. 
		the order is: choice (pre-action), peri-action (centered around the action ts),
		delay (time before the rewarde ts), and reward (time after the rewarde ts).
	-win_size = the size of each window  (in sec)
	-win_step = the time in s over which to slide the window over each epoch interval. 
		win_size = win_step means windows do not overlap. 
	-smooth: whether or not to use gaussian smoothing. Any value > 0 will smooth with
		a kernel of width smooth
Returns:
	-coeffs: average coefficients for each regressor over the population
	-p_significant: an array where each data point is the fraction of units
		with significant correlation coefficients at a particular timepoint
	-epoch_idx: a dictionary reporting which indices in the window (time) dimenstion 
		correspond to which epochs
	-proportions: if False, will return the counts of units rather than the fractions
	TODO: add this feature [-sig_level: the level above which the num_significant values are significant
		(binomial test)] 
"""
def regress_session_epochs(f_behavior,f_ephys,epoch_durations=[2,0.5,1,2],
	win_size=0.5,win_step=0.25,smooth=10,proportions=True):
	sig_level = 0.05 ##value to count as significant
	##get the list of unit names
	f = h5py.File(f_ephys,'r')
	unit_list = [x for x in f.keys() if x.startswith("sig")]
	num_units = len(unit_list)
	f.close()
	##create an arguments list to send to processes
	arglist = [(f_behavior,f_ephys,x,epoch_durations,win_size,win_step,smooth) for x in unit_list]
	##spawn some processes and have them compute the regressions for each unit
	pool = mp.Pool(processes=mp.cpu_count())
	async_result=pool.map_async(mp_regress_unit_epochs,arglist)
	pool.close()
	pool.join()
	data = async_result.get()
	#parse the results
	##the indices will be the same for all
	epoch_idx = data[0][2]
	##create the output arrays
	coeffs = np.zeros(data[0][0].shape)
	perc_sig = np.zeros(data[0][1].shape)
	for i in range(num_units):
		coeffs = coeffs+data[i][0]
		sig = (data[i][1]<=sig_level).astype(float)
		perc_sig = perc_sig + sig
	##now just divide by the total number of units analyzed
	coeffs = coeffs/float(num_units)
	if proportions:
		perc_sig = perc_sig/float(num_units)
	return coeffs,perc_sig,epoch_idx,num_units

"""
A helper function for using multiple processes on regress_unit_epochs. 
	-Inputs:
		-a tuple of arguments to parse into regress_unit_epochs
"""
def mp_regress_unit_epochs(args):
	f_behavior = args[0]
	f_ephys = args[1]
	unit_name = args[2]
	epoch_durations = args[3]
	win_size = args[4]
	win_step = args[5]
	smooth = args[6]
	coeffs,sig_vals,epoch_idx = regress_unit_epochs(f_behavior,f_ephys,
		unit_name,epoch_durations,win_size,win_step,smooth)
	return coeffs,sig_vals,epoch_idx

"""
This function runs a regression analysis on data from one
unit from one session.
Inputs:
	-f_behavior: file path of HDF5 file containing the behavioral timestamps
	-f_ephys: file path to the HDF5 file containing the ephys data
	-unit_name: name of the unit to run the analysis on
	-epoch_durations: a list of the desired durations (in sec) of each behavioral epoch. 
		the order is: choice (pre-action), peri-action (centered around the action ts),
		delay (time before the rewarde ts), and reward (time after the rewarde ts).
	-win_size = the size of each window  (in sec)
	-win_step = the time in s over which to slide the window over each epoch interval. 
		win_size = win_step means windows do not overlap. 
	-smooth: whether or not to use gaussian smoothing. Any value > 0 will smooth with
		a kernel of width smooth
Returns:
	-coeffs: an array of estimated regressor coefficients for each time window (coeff x win)
	-sig_vals: an array of coeff significance values for each time window (corresponds to 
		coeffs)
	-epoch_idx: a dictionary reporting which indices in the window (time) dimenstion 
		correspond to which epochs

******NOTE: I'm building in gaussian smoothing to this function, might want to remove later*****
"""
def regress_unit_epochs(f_behavior,f_ephys,unit_name,
	epoch_durations=[2,0.5,1,2],win_size=0.5,win_step=0.25,smooth=10):
	##start by getting the paired timestamp and regressor data
	ts_data, regressors = ts_and_regressors(f_behavior)
	##now get the spike data for the requested unit
	spike_data = pe.get_spike_data(f_ephys,unit_name,smooth)
	num_trials = regressors.shape[0]
	num_regressors = regressors.shape[1]
	num_windows = pt.get_num_windows(epoch_durations,win_size,win_step)
	##allocate the output arrays
	coeffs = np.zeros((num_regressors,num_windows))
	sig_vals = np.zeros((num_regressors,num_windows))
	##build the data arrays. y_windows[i,:] is the spike rate
	##in window i for every trial in the session
	y_windows = np.zeros((num_windows,num_trials))
	for t in range(num_trials):
		window_edges, epoch_idx = pt.get_epochs(ts_data[t,:],epoch_durations,win_size,win_step)
		##now we want to fill our spike data array with spike rates in each window.
		for w in range(window_edges.shape[0]):
			y_windows[w,t] = pe.window_rate(window_edges[w,:],spike_data)
	##now we have all the data needed to run the regressions and fill
	##the output arrays:
	for win in range(num_windows):
		y = y_windows[win,:]
		X = regressors
		c = run_regression(y,X)
		p = permutation_test(c,y,X)
		coeffs[:,win] = c
		sig_vals[:,win] = p
	##all done, return results
	return coeffs,sig_vals,epoch_idx 

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
		Qu = 0.05
	elif block_type == 'upper_rewarded':
		Qu = 0.85
		Ql = 0.05
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
	-y: a vector of regressands, which is the mean spike rate over 
		an interval of interest for n trials (see get_regressands)
	-X: an n-trials by m-regressors array (see get_regresssors)
Returns:
	-coeff: fitted coefficient values
	-pvals: significance of fitted coeffiecient values based on t-test
"""
def run_regression(y,X):
	##initialize the regression
	regr = linear_model.LinearRegression()
	##fit the model
	regr.fit(X,y)
	##get the coefficients
	coeff = regr.coef_
	return coeff

"""
A function to run a parametric t-test on the regression coefficients.
Inputs:
	y: regressand data
	X: regressor array (n-trials by m-regressors)
Returns:
	F: F-values for each regressor
	p: p-value for each regression coefficient
"""
def t_test_coeffs(y,X):
	F,p = f_regression(X,y)
	return F,p

"""
A function to test the significance of regression coefficients
using a permutation test. A parametric t-test could also be used, but
if action values are included in the regression it makes more sense to 
use permutation tests (see Kim et al, 2009)
Inputs:
	-coeffs: the values of the coefficients to test for significance
	-y: regressand data used to generate the coefficients
	-X: regressor data (trials x features)
	-repeat: number of trials to conduct
Returns:
	p: p-value for each coefficient, defined as the frequency with which
		the shuffled result exceeded the actual result
"""
def permutation_test(coeffs,y,X,repeat=1000):
	regr = linear_model.LinearRegression()
	##the # of times the shuffled val exceeded the experimental val
	c_exceeded = np.zeros(coeffs.size)
	for i in range(repeat):
		y_shuff = np.random.permutation(y)
		regr.fit(X,y_shuff)
		c_shuff = regr.coef_
		for i in range(c_shuff.size):
			if abs(c_shuff[i]) > abs(coeffs[i]):
				c_exceeded[i]+=1
	p = c_exceeded/float(repeat)
	return p

"""
This function is used to generate PAIRED timestamp and regressor data.
I probably could have designed this better, but the sort_by_trial function
breaks the results into two blocks, which is needed to apply the correct regressor
values. But, you want to keep the block data in the same order when you concatenate
all of the timestamps to get time-locked ephys data. this function generates the concatenated
regressor data as well as the concatenated timestamp data such that it's all in the
same order. 
Inputs:
	-f_in: path to the hdf5 file containing the timestamp data
Outputs:
	-regressors: an array of regressors returned by get_regressors
	-ts: concatenated timestamps in the same order as the regressors
"""
def ts_and_regressors(f_in):
	##get the results dictionary
	sorted_data = pt.sort_by_trial(f_in,save_data=False)
	block_names = sorted_data.keys() ##names of the blocks
	ts = None
	regressors = None
	##concatenate both regressors and behavior ts
	##at the same time to preserve order for the next step
	for n in block_names:
		block_data = sorted_data[n]
		regr_data = get_regressors(block_data,n)
		if ts != None:
			ts = np.vstack((ts,block_data))
		else:
			ts = block_data
		if regressors != None:
			regressors = np.vstack((regressors,regr_data))
		else:
			regressors = regr_data
	return abs(ts), regressors




######DEPRECATED FUNCTIONS ###########


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
# def session_regression(f_behavior,f_ephys,window=[2,1,1]):
# 	result = {} ##dictionary to return
# 	#get the behavioral data:
# 	behavior_data, regressors = ts_and_regressors(f_in)
# 	##now work on the ephys data
# 	signals = pe.get_spike_data(f_ephys) ##the binary spike data
# 	intervals = ['pre-action','pre-reward','post-reward']##the three periods to look @
# 	triggers = ['action','outcome','outcome'] #ehhh... this will make sense later
# 	##generate the windows
# 	windows = ([window[0],0],[window[1],0],[0,window[2]])
# 	for n, inter in enumerate(intervals):
# 		##create a nested dictionary for this interval that will go in our output dict
# 		units_dict = {}
# 		##get the corresponding timestamp data to use
# 		trigger_type = triggers[n]
# 		##get the window for this interval
# 		win = windows[n]
# 		##run through each unit and calculate the regression
# 		for sig in signals.keys():
# 			data = signals[sig]
# 			##get the regressand data
# 			y = get_regressand(behavior_data,trigger_type,data,win)
# 			X = regressors
# 			##get the regression data for this unit in this interval
# 			coef,pvals = run_regression(y,X)
# 			##now save this data to the dictionary
# 			units_dict[sig] = (coef,pvals)
# 		##now save this data to the outermost dictionary
# 		result[inter] = units_dict
# 	return result


