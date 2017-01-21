###PCA.py

##functions for assembling data for PCA
##and performing PCA analysis

import parse_ephys as pe
import parse_timestamps as pt
import numpy as np
import plotting as ptt
from scipy.stats import zscore
from numpy import linalg as la
import ml_regress as mr
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

"""
A function to get the data matrix for full trials for a given session. 
The lengths of each trial are variable, so here we want to arrange them 
into a data matrix, where axis 0 is the number of units, and axis 1 is each
concatenated trial. In addition, we want to be able to parse the individual trials.
So included in the output is indices of relevant timestamps, as well as indices that
indicate which condition a given trial falls into.
Inputs:
	-f_behavior: HDF5 file containing behavior timestamps
	-f_ephys: HDF5 file containing ephys data
	-bin: bin size, in ms
	-smooth: width of gaussian kernel to smooth with; 0 is no smoothing
	-pre: time, in seconds, before the action to consider the start of a trial
	-post: time, in seconds, after a reward timestamp to consider as the end of a trial.
Outputs:
	-X: data matrix containing full trials; shape n-units by n-trials x trial lengths
	-trial_index: array of size n-trials x 4 containing the timestamps for each trial.
		in dim 1, the order is trial start, action, outcome, trial end.
	-cond_idx: indices of the various conditions
"""
def data_matrix_full_trials(f_behavior,f_ephys,bins=20,smooth=10,pre=1.5,post=1,plot=True):
	##get the data from the ephys file
	spike_data = pe.bin_spikes_all(f_ephys,bins,smooth=smooth,z=False) ##want to Z-score later
	unit_names = spike_data.keys()
	##now parse the timestamps.
	trial_data,regr,cond_idx = mr.ts_and_regressors2(f_behavior)
	##generate the new timestamp array based on input params
	ts = []
	for i in range(trial_data.shape[0]):
		##check to see if this trial is within the time range
		trial_ts = trial_data[i,:]
		##the new trial timestamp array
		ts.append(np.array([
			trial_ts[1]-pre,
			trial_ts[1],
			trial_ts[2],
			trial_ts[2]+post]))
	ts = np.asarray(ts)
	num_units = len(unit_names)
	num_trials = ts.shape[0]
	##convert timestamps to bins
	ts = pt.ts_to_bins(ts,bins)
	##figure out how long our output array will be, given that we have different trial lengths
	n_total = 0
	for i in range(ts.shape[0]):
		n_total += ts[i,3]-ts[i,0]
	##allocate output matrix
	X = np.zeros((num_units,n_total))
	##fill the matrix with the trial data
	for u in range(num_units):
		cursor = 0 ##index of last data
		signal = spike_data[unit_names[u]]
		for t in range(ts.shape[0]):
			win = [ts[t,0],ts[t,3]]
			trial_data = pe.data_window3(signal,win)
			X[u,cursor:cursor+trial_data.shape[0]] = trial_data
			cursor += trial_data.shape[0]
			#print "cursor is "+str(cursor)
	##NOW we can z-score each row
	for i in range(X.shape[0]):
		X[i,:] = zscore(X[i,:])
	if plot:
		ptt.plot_X_sample2(X,ts)
	return X, ts, regr, cond_idx

"""
A function to build the raw data matrix, X, for a
given session, including all trial conditions (see
pt.sort_trials_by_condition).  
Inputs:
	-f_behavior: HDF5 file containing behavior timestamps
	-f_ephys: HDF5 file containing ephys data
	-epoch: str, containing the name of the epoch (choose one of
		'['choice','action','delay','reward'])
	-epoch_window: [pre,post] event in s
	-bin: bin size, in ms
	-smooth: width of gaussian kernel used to smooth data
Outputs:
	X, a data matrix of N-unit by (c conditions * t bins)
	coeffs, a matrix of regression coefficients for each trial
		included in X.
	cond_idx: a dictionary indicating what trial indices fall
		under what condition
"""
def get_data_matrix(f_behavior,f_ephys,epoch,epoch_window=[1.5,0.5],
	bins=100,smooth=50,plot=True):
	##get the data from the ephys file
	spike_data = pe.bin_spikes_all(f_ephys,bins,smooth=smooth,z=False) ##want to Z-score later
	unit_names = spike_data.keys()
	##now parse the timestamps.
	trial_data,regr,cond_idx = mr.ts_and_regressors2(f_behavior)
	##now get the specific timestamps
	ts = trial_data[:,pt.epoch_LUT(epoch)]
	##allocate the return array. 
	num_units = len(unit_names)
	num_trials = ts.shape[0]
	##we also need to get our timestamps in terms of bin size
	ts = pt.ts_to_bins(ts,bins)
	epoch_window[0] = np.ceil((epoch_window[0]*1000.0)/bins)
	epoch_window[1] = np.ceil((epoch_window[1]*1000.0)/bins)
	epoch_window = np.asarray(epoch_window).astype(int)
	e_bins = int(epoch_window[0]+epoch_window[1])
	X = np.zeros((num_units,(num_trials*e_bins)))
	##get the data for each unit and each trial
	for u in range(num_units):
		signal = spike_data[unit_names[u]]
		idx = 0
		for t in range(ts.shape[0]):
			win = [ts[t]-epoch_window[0],ts[t]+epoch_window[1]]
			trial_data = pe.data_window3(signal,win)
			X[u,idx:idx+e_bins] = trial_data
			idx+=e_bins
	##NOW we can z-score each row
	for i in range(X.shape[0]):
		X[i,:] = zscore(X[i,:])
	if plot:
		ptt.plot_X_sample(X,e_bins)
	return X, regr, cond_idx

"""
A function that uses probabalistic PCA to decompose a
data matrix, X.
Inputs:
	X: data matrix of shape n-units by t-samples
Outputs:
	X_pca: data matrix of n-components by n_units
	var_explained: array containing the ratio of variance explained for each component 
"""
def ppca(X):
	##init the PCA object
	pca = PCA(svd_solver='full',n_components='mle') ##this sets it up to be PPCA
	pca.fit(X.T)
	X_pca = pca.components_
	var_explained = pca.explained_variance_ratio_
	return X_pca, var_explained

"""
A function to compute the covariance matrix of a raw data matrix, X (see above)
Inputs:
	-X: raw data matrix; N-units by (t*b)
Returns:
	-C: covariance matrix of the data
"""
def cov_matrix(X,plot=True):
	C = np.dot(X,X.T)/X.shape[1]
	if plot:
		ptt.plot_cov(C)
	return C

"""
A function that uses spectral decomposition to find the
eigenvalues and eigenvectors of a covariance matrix
Inputs:
	-C: covariance matrix of the data
Returns:
	-w: eigenvalues
	-v: eigenvectors (PC's)
"""
def spectral_decomp(C,X=None,plot=True):
	w,v = la.eig(C)
	##organize them in order of most to least variance captured
	idx = np.argsort(w)[::-1]
	w = w[idx]
	v = v[:,idx].T ##rearrange so the first axis indexes the PC vectors
	if plot:
		ptt.plot_eigen(C,X,w,v)
	return w,v

"""
A function to compute the "denoising matrix" based on the computed
PCs.
Inputs: 
	-v; array of principal components (eigenvectors) arranged 
	in order of variance explained, and i of shape [n vectors x n values]
	-num_PCs: how many PCs to use in the de-noising matrix
Returns: D; de-noising matrix of size N-units by N-units
"""
def get_denoising_matrix(v,num_PCs):
	D = np.dot(v[0:num_PCs,:].T,v[0:num_PCs])
	return D


"""
a population to compute the "denoised" population
response given a denoising matrix and some raw data
Inputs: 
	-X; raw data matrix
	-D: denoising matrix
Returns:
	-Xpca: denoised version of X
"""
def denoise_X(X,D):
	return np.dot(D,X)

"""
a helper function to find the maximum norm of an array
along the LAST AXIS
Inputs:
	A: array; must be 3-D
Returns:
	Amax: maximum absolute value of A along the last axis
"""
def max_norm(A):
	Amax = np.zeros((A.shape[0],A.shape[1]))
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			idx = np.argmax(abs(A[i,j,:]))
			Amax[i,j] = A[i,j,idx]
	return Amax



"""
a helper function to index a data matrix according
to trial indices
Inputs:
	X; data matrix (units x trials)
	trial_len: length, in bins, of trials
	idx: array of trial numbers to take
Outputs: X data array with only the trials of interest remaining
"""
def condition_sort(X,trial_len,index):
	##the output array 
	X_c = np.zeros((X.shape[0],trial_len*index.size))
	for n, idx in enumerate(index):
		X_c[:,n*trial_len:(n+1)*trial_len] = X[:,idx*trial_len:(idx+1)*trial_len]
	return X_c

"""
A function to orthoganalize a matrix
Inputs:
	B: matrix to orthoganalize
Returns:
	Q: orthogonalized matrix
"""
def qr_decomp(B):
	q,r = np.linalg.qr(B,mode='full')
	return q

"""
A function to take a full data matrix, X, containing trials of all different
lengths, and returning the data matrix with the trials stretched and squeezed into
the same shape.
Inputs:
	X: full trial data matrix
	ts: timestamps of trials
Returns:
	Xnew: data matrix with all trials same length; rearranged so axis 0 is units, 
		axis 1 is trials, and axis 2 is time.
	trial_len: new trial length that has been applied to all trials
"""
def equalize_trials(X,ts):
	Xnew = [] ##resulting array to return
	for u in range(X.shape[0]): ##process each unit separately
		u_data = X[u,:] ##all data for this unit
		u_data = split_unit_vector(u_data,ts) ##split into trials in a list
		u_data = interp_trials(u_data) ##adjusted to all the same length as an array
		Xnew.append(u_data)
	return np.asarray(Xnew)



"""
A function to split a data matrix vector for a single unit into individual trials.
Inputs:
	x: data matrix vector for 1 unit (1-d)
	ts: timestamps of trials, in bins
Returns:
	trials: a list of arrays, where each list element is the data from one trial
"""
def split_unit_vector(x,ts):
	trials = []
	cursor = 0
	for t in range(ts.shape[0]):
		##transform the timestamps to be relative to the start of each trial
		ts[t,:] = ts[t,:]-ts[t,0]
		##get the trial data
		trial_len = ts[t,3]
		trial_data = x[cursor:cursor+trial_len]
		trials.append(trial_data)
	return trials

"""
A function to stretch individual trials of different lengths 
so that each trial is the same length
Inputs:
	trials: a list containing vectors of varying lengths from different trials
Returns:
	trials: an array where vectors have been interpolated to be the mean length of all vectors
"""
def interp_trials(trials):
	##figure out the mean trial length
	new_trials = []
	mean_len = 0
	for i in range(len(trials)):
		mean_len+=trials[i].size
	mean_len = np.ceil(float(mean_len)/len(trials)).astype(int)
	#print "mean_len = "+str(mean_len)
	##decide if this trial needs additional points, or needs points removed
	for t in range(len(trials)):
		data = trials[t]
		while data.shape[0] > mean_len: ##case where the trial is longer
			diff = data.shape[0]-mean_len
			##generate randomly some points to remove to make it the right len
			rmv = np.random.randint(1,high=mean_len,size=diff)
			data = np.delete(data,rmv,axis=0)
		if data.shape[0] < mean_len: ##case where the trial is shorter
			diff = mean_len-data.size
			##generate randomly some indices to add and interpolate to
			add = np.random.randint(0,high=data.size-1,size=diff)
			data = np.insert(data,add,np.nan,axis=0)
			not_nan = np.logical_not(np.isnan(data))
			indices = np.arange(len(data))
			data = np.interp(indices,indices[not_nan],data[not_nan])
			new_trials.append(data)
		else:
			new_trials.append(data)
	new_trials = np.asarray(new_trials)
	return new_trials

"""
A function to separate out trials of different lengths
AND sort them by condition from a data matrix, so each trial has
data from all units
Inputs:
	X: full data trial matrix
	ts: timestamps of the various trials
	cond_idx: dictionary with indices of trials split according to condition
Returns:
	trials_dict: dictionary, where keys are the same keys as cond_idx,
		and the values are lists with the trials separated by condition
	ts_dict: the relative timestamps for every trial, also sorted
"""
def sort_full_trials(X,ts,cond_idx):
	##get the timstamps relative to the data matrix
	ts_rel = reorder_ts(ts)
	##create the output dictionary
	trials_dict = {}
	ts_dict = {}
	##process each condition separately
	for key in cond_idx.keys():
		trial_idx = cond_idx[key] ##the trial indices for this condition
		cond_ts = ts_rel[trial_idx,:] ##the indices in X of the trials for this condition
		trials = [] ##the list of trials to return
		for t in range(cond_ts.shape[0]):
			trial_data = X[:,cond_ts[t,0]:cond_ts[t,3]]
			trials.append(trial_data)
		trials_dict[key] = trials
		ts_dict[key] = cond_ts
	return trials_dict,ts_dict

"""
A helper function to reorder timestamps so that they index
a data matrix, rather than the continuous data from a full trial
Inputs:
	ts: timestamps array
Returns: 
	ts_rel: timetamps with values relative to a data matrix
"""
def reorder_ts(ts):
	ts_rel = np.zeros(ts.shape)
	cursor = 0 ##keep track of where we are
	for trial in range(ts.shape[0]):
		times = ts[trial,:] ##the original ts for this trial
		times = times-times[0] ##now the times are relative to the start of the trial
		times = times+cursor ##now the times are relative to the start of the last trial
		ts_rel[trial,:] = times ##add data to the return array
		cursor = times[3] ##the end of this trial
	return ts_rel

"""
A function to plot the mean, interpolated results of a series of 
different length trials projected onto two different task-relevant axes
Inputs:
	trials_list: a list of trials (of different lenghths, containing data from all
		units arranged units x time in trial)
	x_vec: the orthogonalized x-vector to project the data onto; size n-units
	y_vec: same, but for the y-axis 
Returns:
	x-vals: mean x values of the projection
	y-vals: mean y-vals
	mean_ts: mean of the behavioral timestamps
	std_ts: the standard deviation of the ts vals
"""
def project_full_trials(trials_list,x_vec,y_vec,ts):
	##first, start by projecting all of the trials onto the given vectors
	x_projs = [] ##the x-projections for all trials
	y_projs = [] ##the y-projections for all trials
	for t in range(len(trials_list)):
		data = trials_list[t] ##the data for this trial
		##project the data onto the two vectors and add to the lists
		x_projs.append(np.dot(x_vec,data))
		y_projs.append(np.dot(y_vec,data))
	##now we have the data projections, 
	##and we want to squish or stretch them to be the same size
	x_projs = interp_trials(x_projs)
	y_projs = interp_trials(y_projs)
	##now get the mean and std of the behavioral ts
	##first make them all relative to each other
	for i in range(ts.shape[0]):
		ts[i,:] = ts[i,:]-ts[i,0]
	mean_ts = ts.mean(axis=0)
	
	std_ts = ts.std(axis=0)
	return x_projs, y_projs,mean_ts,std_ts