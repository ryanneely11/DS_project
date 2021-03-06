##parse_ephys.py
##this function works with trial timestamps generated by 
##parse_timestamps.py. It splits whole session ephys data
##into individual trials, as well as individual behavioral events

import os
import glob
import numpy as np
#import plxread
import h5py
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import zscore

####TODO: plxread not install correctly!######



"""
this script looks in a directory, takes the plx files and saves a copy as an HDF5 file.

"""
def batch_plx_to_hdf5(directory):
	##first, get a list of the plx files in the directory:
	cd = os.getcwd() ##to return to the cd later
	os.chdir(directory)
	for f in glob.glob("*.plx"):
		cur_file = os.path.join(directory,f)
		print "Saving "+cur_file
		##parse the plx file
		data = plxread.import_file(cur_file,AD_channels=range(1,97),import_unsorted=False,
			verbose=False,import_wf=True)
		##create the output file in the same dir
		out_file = h5py.File(cur_file.strip('plx')+'hdf5','w-')
		##save the data
		for k in data.keys():
			out_file.create_dataset(k,data=data[k])
	out_file.close()
	os.chdir(cd)
	print "Done!"
	return None

"""
This function returns a dictionary of spike data, in binary form, 
given an input hdf5 file.
Inputs:
	-f_in: an hdf5 file with the ephys data
	-smooth: any value > 0 will smooth with a gaussian kernel of width smooth
Outputs:
	-result: a dictionary where the keys are unit names and the values
		are binary arrays of the spike data
"""
def get_spike_data_all(f_in,smooth=0):
	##dict to return 
	result = {}
	##get the duration of this session in ms
	duration = get_session_duration(f_in)
	##open our source file
	f = h5py.File(f_in,'r')
	##get the names of all the sorted units contained in this file
	units_list = [x for x in f.keys() if x.startswith("sig")]
	f.close()
	##process each of these sorted units, and add the data to the result dict
	for unit in units_list:
		signal = get_spike_data(f_in,unit,smooth)
		##add to the result
		result[unit] = signal
	return result

"""
This function returns the binary transformed spike data
for one unit.
Inputs:
	-f_in: HDF5 data file to get the data from
	-unit_name: name of the unit to get data for
	-smooth: a value of smooth > 0 smooths with a gaussian kernel of width smooth
returns:
	-result: binary transmormed data array
"""
def get_spike_data(f_in,unit_name,smooth=0):
	duration = get_session_duration(f_in) ##duration of this session
	##open the source file and get the requested unit data
	f = h5py.File(f_in,'r')
	spike_ts = np.asarray(f[unit_name])
	f.close()
	##do the transform
	result = pt_times_to_binary(spike_ts,duration,smooth)
	##return the result
	return result


"""
This script takes in a 1-D array of timestamps, a single ephys signal (1-D)
and returns the windowed data around the timestamps.
Input:
	-1-D array of timstamps (in sec)
	-1-D array of signal (sampled in ms)
	-Window size (in sec)
Output: 
	-an x-timestamps by n ms array of the windowed data
	-a 1-D array that indicates the indices (if any) of the windows
		/ trials contains no data
"""
def data_windows(timestamps,signal,window=[3,3]):
	##get rid of singleton dims
	timestamps = np.squeeze(timestamps)
	signal = np.squeeze(signal)
	##pad the data so you can take windows at the edge
	pad_1 = np.zeros(abs(window[0])*1000)
	pad_2 = np.zeros(abs(window[1])*1000)
	signal = np.hstack((pad_1,signal,pad_2))
	##shift the timestamps to account for padding
	timestamps = timestamps+window[0]
	##allocate memory
	result = np.zeros((timestamps.size,(window[0]*1000+window[1]*1000)))
	for i in range(timestamps.size):
		start = np.ceil((timestamps[i]*1000.0-window[0]*1000.0)).astype(int)
		stop = np.ceil((timestamps[i]*1000.0+window[1]*1000.0)).astype(int)
		idx = np.arange(start,stop)
		result[i,:] = signal[idx]
	##now check if there are any empty trials
	row_total = result.sum(axis=1)
	no_data = np.where(row_total==0)
	return result,no_data

"""
This function is similar to the above, but instead of taking a center
timestamp to window around, it takes two edges to take data in between.
Inputs:
	-endpoints: the endpoints of the data window (in sec)
	-signal: a binary-transformed spike data array
Returns:
	-result: the requested data window
"""
def data_window2(endpoints,signal):
	##make sure the data is pretty
	signal = np.squeeze(signal)
	endpoints = np.asarray(endpoints) ##in secs right here
	start = np.ceil((endpoints[0]*1000.0)).astype(int) #convert to ms/index
	stop = np.ceil((endpoints[1]*1000.0)).astype(int)
	idx = np.arange(start,stop)
	try:
		result = signal[idx]
	except IndexError: ##case where endpoints overrun the bounds of the signal
		signal2 = np.hstack((np.zeros(10000),signal,np.zeros(10000))) ##pad the signal with some zeros
		##adjust the endpoints accordingly
		start = start + 10000
		stop = stop + 10000
		idx = np.arange(start,stop)
		result = signal2[idx]
	return result

"""
Another data windowing function. This one doesn't assume a bin 
size for the input array, so the window is expressed in terms of bins.
Inputs:
	-signal: the full signal to operate on
	-window: the window to take data; expressed in terms of array indices
Output:
	-data: the requested window of data
"""
def data_window3(signal,window):
	##pretty up the data
	signal = np.squeeze(signal)
	start = np.ceil(window[0]).astype(int)
	stop = np.ceil(window[1]).astype(int)
	idx = np.arange(start,stop)
	try:
		result = signal[idx]
	except IndexError: #case where indices overrun the bounds of the data
		signal2 = np.hstack((np.zeros(1000),signal,np.zeros(1000)))
		##adjust the endpoints
		start = start +1000
		stop = stop + 1000
		idx = np.arange(start,stop)
		result = signal2[idx]
	return result
"""
This function is basically an extension of data_window2
that just calculates the spike RATE in that window, so the 
outcome is just some value.
Inputs:
Inputs:
	-endpoints: the endpoints of the data window (in sec)
	-signal: a binary-transformed spike data array
Returns:
	-result: the spike rate over the data window
"""
def window_rate(endpoints,signal):
	data = data_window2(endpoints,signal)
	##figure out how long the window is (in sec)
	Nwin = float(endpoints[1]-endpoints[0])
	datarate = data.sum()/Nwin
	return datarate

"""
This function returns a dictionary with all the data windows
around ONE set of timestamps for ALL sorted neurons in a file.
Inputs:
	-f_in: HDF5 file containing ephys data
	-timestamps: timestamps to lock to 
	-window_size: list, with [-3,3] corresponding to 3 secs before and 3 after
Outputs:
	-result: dictionary containing the windows, where the keys are 
		the names of the individual units
"""
def data_windows_multi(f_in,timestamps,window=[3,3]):
	##dict to return 
	result = {}
	##get the duration of this session in ms
	duration = get_session_duration(f_in)
	##open our source file
	f = h5py.File(f_in,'r')
	##get the names of all the sorted units contained in this file
	units_list = [x for x in f.keys() if x.startswith("sig")]
	##process each of these sorted units, and add the data to the result dict
	for unit in units_list:
		signal = np.asarray(f[unit])
		##convert the signal to a binary array
		signal = pt_times_to_binary(signal,duration)
		##get the data windows for this unit
		windows,no_data = data_windows(timestamps,signal,window)
		##add to the result
		result[unit] = windows
	f.close()
	return result



"""
a helper function to convert spike times to a binary array
ie, an array where each bin is a ms, and a 1 indicates a spike 
occurred and a 0 indicates no spike
Inputs:
	-signal: an array of spike times in s(!)
	-duration: length of the recording in ms(!)
	-smooth: if smooth > 0, use a gaussian kernel 
		to smooth the binned spikes, with a kernel width of
		whatever smooth ms
Outputs:
	-A duration-length 1-d array as described above
"""
def pt_times_to_binary(signal,duration,smooth=0):
	##convert the spike times to ms
	signal = signal*1000.0
	##get recodring length
	duration = float(duration)
	##set the number of bins as the next multiple of 100 of the recoding duration;
	#this value will be equivalent to the number of milliseconds in 
	#the recording (plus a bit more)
	numBins = int(np.ceil(duration/100)*100)
	##do a little song and dance to ge the spike train times into a binary format
	bTrain = np.histogram(signal,bins=numBins,range=(0,numBins))
	bTrain = bTrain[0].astype(bool).astype(int)
	if smooth > 0:
		bTrain = gauss_convolve(bTrain,smooth)
	bTrain = zscore(bTrain)
	return bTrain

"""
a function that takes in an array of spike timestamps, 
in bins them into arbitrary size bins.
Inputs:
	-f_in: path to data file
	-unit_name: name of unit to get data for in the file 
	-bin_size: size of bins, in ms
	-smooth: whether to smooth with gaussian kernel of width smooth. 0 is no smoothing
	-z: if true, returns the z-score of the data
Outputs:
	-binned_spikes
"""
def bin_spikes(f_in,unit_name,bin_size,smooth=0,z=True):
	##get the duration of the recording
	duration = float(get_session_duration(f_in))	
	##get the spike data
	f = h5py.File(f_in,'r')
	##convert timestamps to ms
	signal = np.asarray(f[unit_name])*1000.0
	f.close()
	##set the functional duration as the next multiple of 100 of the recoding duration;
	#this value will be equivalent to the number of milliseconds in 
	#the recording (plus a bit more)
	duration = np.ceil(duration/100)*100
	##get the number of bins
	num_bins = int(np.ceil(duration)/float(bin_size))
	binned_spikes = np.histogram(signal,bins=num_bins,range=(0,duration))[0]
	if smooth > 0:
		binned_spikes = gauss_convolve(binned_spikes,smooth)
	if z:
		binned_spikes = zscore(binned_spikes)
	return binned_spikes

"""
A function to bin all spikes from a session. 
Inputs:
	-f_in: file path of ephys data
	-bin_size: size of bins in ms
	-smooth: whether to smooth with gaussian kernel of width smooth. 0 is no smoothing
	-z: if true, returns the z-score of the data
"""
def bin_spikes_all(f_in,bin_size,smooth=0,z=True):
	##dict to return 
	result = {}
	##open our source file
	f = h5py.File(f_in,'r')
	##get the names of all the sorted units contained in this file
	units_list = [x for x in f.keys() if x.startswith("sig")]
	f.close()
	##process each of these sorted units, and add the data to the result dict
	for unit in units_list:
		signal = bin_spikes(f_in,unit,bin_size,smooth,z)
		##add to the result
		result[unit] = signal
	return result

"""
A helper function to get the duration of a session.
Operates on the principal that the session duration is
equal to the length of the LFP (slow channel, A/D) recordings 
Inputs:
	-file path of an hdf5 file with the ephys data
Outputs:
	-duration of the session in ms(!), as an integer rounded up
"""
def get_session_duration(f_in):
	f = h5py.File(f_in, 'r')
	##get a list of the LFP channel timestamp arrays
	##(more accurate than the len of the value arrs in cases where
	##the recording was paused)
	AD_ts = [x for x in f.keys() if x.endswith('_ts')]
	##They should all be the same, so just get the first one
	sig = AD_ts[0]
	duration = np.ceil(f[sig][-1]*1000.0).astype(int)
	f.close()
	return duration

"""
A function to convolve data with a gaussian kernel of width sigma.
Inputs:
	array: the data array to convolve. Will work for multi-D arrays;
		shape of data should be samples x trials
	sigma: the width of the kernel, in samples
"""
def gauss_convolve(array, sigma):
	##remove singleton dimesions and make sure values are floats
	array = array.squeeze().astype(float)
	##allocate memory for result
	result = np.zeros(array.shape)
	##if the array is 2-D, handle each trial separately
	try:
		for trial in range(array.shape[1]):
			result[:,trial] = gaussian_filter(array[:,trial],sigma=sigma,order=0,
				mode="constant",cval = 0.0)
	##if it's 1-D:
	except IndexError:
		if array.shape[0] == array.size:
			result = gaussian_filter(array,sigma=sigma,order=0,mode="constant",cval = 0.0)
		else:
			print "Check your array input to gaussian filter"
	return result


"""
A function to get all spike data for one session and compile it into a single
data matrix.
Inputs:
	f_ephys: address of the hdf5 file to get session data from
	z: if True, runs zscore on the full array.
	bin_size: size of bins to use for spike counts (1 = binary array)
	smooth: if >0, smooths data with a gaussian kernel of smooth width
Returns:
	spike_matrix: of size units x bins for the session
"""
def session_unit_matrix(f_ephys,Z=True,bin_size=20,smooth=12):
	##get the unit data in dictionary form
	unit_dict = bin_spikes_all(f_ephys,bin_size,smooth=smooth,z=Z)
	##now just concatenate everything into an array
	spike_matrix = np.zeros((len(unit_dict.keys()),unit_dict[unit_dict.keys()[0]].size))
	for i,u in enumerate(unit_dict.keys()):
		spike_matrix[i,:] = unit_dict[u]
	return spike_matrix
