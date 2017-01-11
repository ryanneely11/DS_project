###PCA.py

##functions for assembling data for PCA
##and performing PCA analysis

import parse_ephys as pe
import parse_timestamps as pt
import numpy as np
import plotting as ptt
from scipy.stats import zscore
from numpy import linalg as la

"""
A function to build the raw data matrix, X, for a
given session, including all trial conditions (see
pt.sort_trials_by_condition).  
Inputs:
	-f_behavior: HDF5 file containing behavior timestamps
	-f_ephys: HDF5 file containing ephys data
	-epoch: str, containing the name of the epoch (choose one of
		'['choice','action','delay','reward'])
	-condition: str specifying which type of trials to look at. Choose from:
		['rewarded','unrewarded','upper_lever','lower_lever','upper_rewarded','lower_rewarded']
		alternatively, you can pass 'all,' which will do all conditions
	-epoch_window: [pre,post] event in s
	-bin: bin size, in ms
	-smooth: width of gaussian kernel used to smooth data
Outputs:
	X, a data matrix of N-unit by (c conditions * t bins)
"""
def get_data_matrix(f_behavior,f_ephys,epoch,condition,epoch_window=[1.5,0.5],
	bins=100,smooth=50,plot=True):
	##get the data from the ephys file
	spike_data = pe.bin_spikes_all(f_ephys,bins,smooth=smooth,z=False) ##want to Z-score later
	unit_names = spike_data.keys()
	##now parse the timestamps.
	if condition == 'all':
		d = pt.sort_by_trial(f_behavior)
		trial_data = np.vstack((d['upper_rewarded'],d['lower_rewarded']))
	else:
		trial_data = pt.sort_trials_by_condition(f_behavior)[condition] #will by only the ts for this condition
	##now get the specific timestamps
	ts = abs(trial_data[:,pt.epoch_LUT(epoch)])
	##allocate the return array. 
	num_units = len(unit_names)
	num_trials = ts.shape[0]
	##we also need to get our timestamps in terms of bin size
	ts = pt.ts_to_bins(ts,bins)
	epoch_window[0] = np.ceil((epoch_window[0]*1000.0)/bins)
	epoch_window[1] = np.ceil((epoch_window[0]*1000.0)/bins)
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
	return X

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

