import h5py
import numpy as np


"""
Takes in as an argument a log.txt files
Returns a dictionary of arrays containing the 
timestamps of individual events
"""
def parse_log(f_in):
	##open the file
	f = open(f_in, 'r')
	##set up the dictionary
	results = {
	"top_rewarded":[],
	"bottom_rewarded":[],
	"trial_start":[],
	"session_length":0,
	"reward_primed":[],
	"reward_idle":[],
	"top_lever":[],
	"bottom_lever":[],
	"rewarded_poke":[],
	"unrewarded_poke":[]
	}
	##run through each line in the log
	label, timestamp = read_line(f.readline())
	while  label is not None:
		#print timestamp
		##now just put the timestamp in it's place!
		if label == "rewarded=top_lever":
			results['top_rewarded'].append(float(timestamp))
		elif label == "rewarded=bottom_lever":
			results['bottom_rewarded'].append(float(timestamp))
		elif label == "trial_begin":
			results['trial_start'].append(float(timestamp))
		elif label == "session_end":
			results['session_length'] = [float(timestamp)]
		elif label == "reward_primed":
			results['reward_primed'].append(float(timestamp))
		elif label == "reward_idle":
			results['reward_idle'].append(float(timestamp))
		elif label == "top_lever":
			results['top_lever'].append(float(timestamp))
		elif label == "bottom_lever":
			results['bottom_lever'].append(float(timestamp))
		elif label == "rewarded_poke":
			results['rewarded_poke'].append(float(timestamp))
		elif label == "unrewarded_poke":
			results['unrewarded_poke'].append(float(timestamp))
		else:
			print "unknown label: " + label
		##go to the next line
		label, timestamp = read_line(f.readline())
	f.close()
	return results

##a sub-function to parse a single line in a log, 
##and return the timestamp and label components seperately
def read_line(string):
	label = None
	timestamp = None
	if not string == '':
		##figure out where the comma is that separates
		##the timestamp and the event label
		comma_idx = string.index(',')
		##the timestamp is everything in front of the comma
		timestamp = string[:comma_idx]
		##the label is everything after but not the return character
		label = string[comma_idx+1:-1]
	return label, timestamp

##takes in a dictionary created by the log_parse function and 
##saves it as an hdf5 file
def dict_to_h5(d, path):
	f_out = h5py.File(path, 'w-')
	##make each list into an array, then a dataset
	for key in d.keys():
		##create a dataset with the same name that contains the data
		f_out.create_dataset(key, data = np.asarray(d[key]))
	##and... that's it.
	f_out.close()

##converts all the txt logs in a given directory to hdf5 files
def batch_log_to_h5(directory):
	log_files = get_log_file_names(directory)
	for log in log_files:
		##generate the dictionary
		result = pt.parse_log(log)
		##save it as an hdf5 file with the same name
		new_path = os.path.splitext(log)[0]+'.hdf5'
		dict_to_h5(result, new_path)
	print 'Save complete!'

##offsets all timestamps in a log by a given value
##in a h5 file like the one produced by the above function
def offset_log_ts(h5_file, offset):
	f = h5py.File(h5_file, 'r+')
	for key in f.keys():
		new_data = np.asarray(f[key])+offset
		f[key][:] = new_data
	f.close()


""" a function to split behavior timestamp data into individual trials.
	Inputs:
		f_in: the file path pointing to an hdf5 file containing
		the behavior event time stamps
		save_data: bool, indicates whether to save the result or not
	Returns: 
		Two n-trial x i behavioral events-sized arrays.
		One contains all trials where the lower lever is rewarded;
		The other contains all the trials where the upper lever is rewarded. 
		Behavioral events are indexed temporally:
		0-trial start; 1-action (U or L); 2-poke(R or U)
"""
def sort_by_trial(f_in,save_data=False):
	#load data file
	f = h5py.File(f_in,'r')
	#get the arrays of timestamps into a dictionary in memory
	data_dict = {
		'lower_lever':np.asarray(f['bottom_lever']),
		'upper_lever':np.asarray(f['top_lever']),
		'reward_idle':np.asarray(f['reward_idle']),
		'reward_primed':np.asarray(f['reward_primed']),
		'rewarded_poke':np.asarray(f['rewarded_poke']),
		'unrewarded_poke':np.asarray(f['unrewarded_poke']),
		'trial_start':np.asarray(f['trial_start']),
		'session_end':np.asarray(f['session_length']),
		'lower_rewarded':np.asarray(f['bottom_rewarded']),
		'upper_rewarded':np.asarray(f['top_rewarded']),
	}
	f.close()
	##create the output dictionary
	result = {}
	##get the dictionary containing the block information
	block_times = get_block_times(data_dict['lower_rewarded'],data_dict['upper_rewarded'],
		data_dict['session_end'])
		##start with all of the lower lever blocks
	try:	
		for lb in range(len(block_times['lower'])):
			block_data = get_block_data(block_times['lower'][lb],data_dict)
			trial_times = sort_block(block_data)
			##case where there is a dictionary entry
			try:
				result['lower_rewarded'] = np.vstack((result['lower_rewarded'],trial_times))
			except KeyError:
				result['lower_rewarded'] = trial_times
	except KeyError:
		pass
	##repeat for upper lever blocks
	try:
		for ub in range(len(block_times['upper'])):
			block_data = get_block_data(block_times['upper'][ub],data_dict)
			trial_times = sort_block(block_data)
			##case where there is a dictionary entry
			try:
				result['upper_rewarded'] = np.vstack((result['upper_rewarded'],trial_times))
			except KeyError:
				result['upper_rewarded'] = trial_times
	except KeyError:
		pass
	if save_data:
		f_out = f_in.strip('hdf5')+'_trial_times.hdf5'
		f = h5py.File(f_out,'w-')
		for key in result.keys():
			f_out.create_dataset(key,data=result[key])
		f.close()
	else:
		return result



"""
A helper function for sort_by_trial; sorts out trials for one block.
Inputs:
	-block_data: a dictionary containing all the timestamps
	for one block, which is a period of time in which the lever-
	reward contingency is constant.
Returns:
	an n-trial by i behavioral events-sized array

	****													****
	****Important: in order to simplify the output arrays, 	****
	****I'm going to use a positive/negative code in the following way:
		For each trial:
			-Timestamp 0 = the start of the trial
			-Timestamp 1 = action; negative number = lower lever; positive = upper lever
			-Timestamp 2 = outcome; negative number = unrewarded; positive = rewarded
"""
def sort_block(block_data):
	##let's define the order of events that we want:
	ordered_events = ['start','action','outcome']
	##allocate memory for the result array
	result = np.zeros((block_data['trial_start'].size,3))
	##fill the results array for each trial
	for i in range(result.shape[0]):
		trial_start = block_data['trial_start'][i] ##the start of this trial
		try: 
			trial_end = block_data['trial_start'][i+1]
		except IndexError:
			trial_end = max(block_data['rewarded_poke'].max(),
							block_data['unrewarded_poke'].max())
		##***ACTIONS***

		##now find the first action
		#idx of any upper presses in the interval
		upper_idx = np.nonzero(np.logical_and(block_data['upper_lever']>trial_start,
			block_data['upper_lever']<trial_end))[0]
		lower_idx = np.nonzero(np.logical_and(block_data['lower_lever']>trial_start,
			block_data['lower_lever']<trial_end))[0]
		##case 1: both upper and lower lever presses happened
		if upper_idx.size>0 and lower_idx.size>0:
			##find which action happened first
			upper_presses = block_data['upper_lever'][upper_idx] ##the actual timestamps
			lower_presses = block_data['lower_lever'][lower_idx]
			##if the first upper press happened first:
			if upper_presses.min()<lower_presses.min():
				action = upper_presses.min()
			elif lower_presses.min()<upper_presses.min():
				action = -1*lower_presses.min()
			else:
				##error case
				print "something wrong in upper/lower comparison"
				break
		#case 2: only upper lever was pressed
		elif upper_idx.size>0 and lower_idx.size==0:
			action = block_data['upper_lever'][upper_idx].min()
		##case 3: only lower lever was pressed
		elif upper_idx.size==0 and lower_idx.size>0:
			action = -1*block_data['lower_lever'][lower_idx].min()
		##case 4: something is wrong!
		else:
			print "Error- no action for this trial??"
			break
		
		##***OUTCOMES***
		
		##ts of any rewarded pokes
		reward_idx = np.nonzero(np.logical_and(block_data['rewarded_poke']>trial_start,
			block_data['rewarded_poke']<=trial_end))[0]
		##case where this was a rewarded trial
		if reward_idx.size == 1:
			outcome = block_data['rewarded_poke'][reward_idx]
		##case where this was not a rewarded trial
		elif reward_idx.size == 0:
			##let's get the unrewarded ts
			unreward_idx = np.nonzero(np.logical_and(block_data['unrewarded_poke']>trial_start,
				block_data['unrewarded_poke']<=trial_end))[0]
			if unreward_idx.size > 0:
				unrewarded_pokes = block_data['unrewarded_poke'][unreward_idx]
				outcome = -1*unrewarded_pokes.min()
			else:
				print "Error: no pokes for this trial"
				break
		else:
			print "error: too many rewarded pokes for this trial"
			break

		##now add the data to the results
		result[i,0] = trial_start
		result[i,1] = action
		result[i,2] = outcome
	return result


"""
A helper function for sort_by_trial; determines how many blocks
are in a file, and where the boundaries are. 
Inputs:
	-Arrays containing the lower/upper rewarded times, 
	as well as the session_end timestamp.
	-min_length is the cutoff length in secs; any blocks shorter 
		than this will be excluded
Outputs:
	A dictionary for each type of block, where the item is a list of 
	arrays with start/end times for that block.
"""
def get_block_times(lower_rewarded, upper_rewarded,session_end,min_length=5*60):
	##get a list of all the block times, and a corresponding list
	##of what type of block we are talking about
	block_starts = []
	block_id = []
	for i in range(lower_rewarded.size):
		block_starts.append(lower_rewarded[i])
		block_id.append('lower')
	for j in range(upper_rewarded.size):
		block_starts.append(upper_rewarded[j])
		block_id.append('upper')
	##sort the start times and ids
	idx = np.argsort(block_starts)
	block_starts = np.asarray(block_starts)[idx]
	block_id = np.asarray(block_id)[idx]
	result = {}
	##fill the dictionary
	for b in range(block_starts.size):
		##range is from the start of the current block
		##to the start of the second block, or the end of the session.
		start = block_starts[b]
		try:
			stop = block_starts[b+1]
		except IndexError:
			stop = session_end
		##check to make sure this block meets the length requirements
		if stop-start > min_length:
			rng = np.array([start,stop])
			##create the entry if needed
			try:
				result[block_id[b]].append(rng)
			except KeyError:
				result[block_id[b]] = [rng]
		else:
			print "Block length only "+str(stop-start)+" secs; excluding"
	return result


"""
another helper function. This one takes in a block start, stop
time and returns only the subset of timestamps that are within
that range.
Inputs:
	block_edges: an array [start,stop]
	data_dict: the dictionary of all the different timestamps
Outputs:
	a modified data dictionary with only the relevant data
"""
def get_block_data(block_edges,data_dict):
	result = {} #dict to return
	keys = data_dict.keys()
	for key in keys:
		data = data_dict[key]
		idx = np.nonzero(np.logical_and(data>block_edges[0],data<=block_edges[1]))[0]
		result[key] = data[idx]
	##figure out if the last trial was completed; if not get rid of it
	last_trial = result['trial_start'].max()
	##need some error catching here in case there were no upper or lower levers in this block
	try:
		last_upper = result['upper_lever'].max()
	except ValueError: ##case of empty array
		last_upper = np.array([0])
	try:
		last_lower = result['lower_lever'].max()
	except ValueError:
		last_lower = np.array([0])
	last_action = max(last_upper,last_lower)
	last_poke = max(result['rewarded_poke'].max(),result['unrewarded_poke'].max())
	if (last_trial < last_action) and (last_trial < last_poke):
		pass
	else:
		result['trial_start'] = np.delete(result['trial_start'],-1)
	return result


"""
a helperfunction for get_epochs. There isn't really a clean or consistant way
to get the windows for each epoch, because their relationship to the actual timestamp
is different for each one. So we have to do a messy if/elif operation.
Inputs:
	-ts: the actual timestamp of the event of interest
	-epoch_type: str, taken from the keys of "periods" in the parent func.
		This will determine how we take windows relative to the timestamp.
	-epoch_duration: in s, the length of this epoch
	-win_size: window len in s
	-win_step: time period by which to advance the windows
"""
def get_windows(ts,epoch_type,epoch_duration,win_size,win_step):
	if epoch_type == "choice":
		start = ts - epoch_duration ##time leading up to action
		stop = ts
	elif epoch_type == "action":
		start = ts - (epoch_duration/2.0) ##time around action
		stop = ts + (epoch_duration/2.0)
	elif epoch_type == "delay":
		start = ts - epoch_duration ##time leading up to reward
		stop = ts
	elif epoch_type == "reward":
		start = ts ##time after reward
		stop = ts + epoch_duration
	##now get the windows
	windows = []
	winstart = start
	winstop = start+win_size
	while winstop <= stop:
		windows.append(np.array([winstart,winstop]))
		winstart+=win_step
		winstop+=win_step
	return np.asarray(windows)

"""
A function to get the windows of all epochs for a given trial.
This function takes in a row of the block data (ie one trial)
and returns all of the epochs alinged, as well as the 
epoch edges
Inputs:
	-trial_ts: timestamps of a single trial (one row of block data)
	-duration of each epoch (s), in order of choice, action, delay, reward
	-window size in s
	-window step size in s
Outputs:
	-results: array of all the windows
	-epoch_idx: which windows belong to which epoch (dict of window indexes)
"""
def get_epochs(trial_ts,durations,win_size,win_step):
	epoch_keys = ['choice','action','delay','reward'] ##epoch periods
	ts_idx = [1,1,2,2] ##index of the timestamps used to locate the above epochs
	##data to return
	results = []
	epoch_idx = {}
	last_idx = 0
	for key, idx, dur in zip(epoch_keys,ts_idx,durations):
		ts = trial_ts[idx] ##timestamp to use
		epoch_windows = get_windows(ts,key,dur,win_size,win_step) ##window data for this epoch
		results.append(epoch_windows)
		epoch_idx[key] = np.arange(len(epoch_windows))+last_idx
		last_idx+=len(epoch_windows)
	return np.vstack(results), epoch_idx

"""
A helper function to calclulate the number of windows to expect
given a list of epoch durations, a window size, and a window step
(all in sec)
Inputs:
	-durations: duration of each epoch (s), in order of choice, action, delay, reward
	-win_size: window size in s
	-win_step: window step size in s
Outputs: the total numbe of windows (int)
"""
def get_num_windows(durations,win_size,win_step):
	num_wins = 0
	for d in durations:
		winstart = np.arange(0,(d-win_size)+win_step,win_step) ##array of all the window starting values
		num_wins += winstart.shape[0] ##number of total windows
	return num_wins
		

"""
a helper function to get a-o data from RL modeling functions
Inputs:
	f_behavior: HDF5 file with behavioral data
Outputs:
	actions: 1-D array of actions; 1 = upper lever; -1 = lower lever
	outcomes: 1-D array of outcomes; 1 = rewarded; -1 = unrewarded
	Qu: action value of upper lever (actual)
	Ql: action value of lower lever (actual)
"""
def get_rl_data(f_behavior):
	pass

####PLOTTING HELPER FUNCTIONS####

##gets info about the duration of trials
def get_trial_durations(data):
	#trial duration is defined as the time between
	#the start of the trial and the first poke
	result = np.zeros(data.shape[0])
	for i in range(result.shape[0]):
		result[i]=abs(data[i,2])-data[i,0] ##abs because negative used to indicate unrewarded
		##encode the outcome in the same manner
		if data[i,2] < 0:
			result[i] = result[i]*-1
	return result

##get the time between the action and the outcome (checking the nosepoke) 
def get_ao_interval(data):
	##diff between the lever and the nosepoke
	result = np.zeros(data.shape[0])
	for i in range(result.shape[0]):
		result[i]=abs(data[i,2])-abs(data[i,1])
		##encode the outcome
		if data[i,2] < 0:
			result[i] = -1.0*result[i]
	return result

def gauss_convolve(array, sigma):
	"""
	takes in an array with dimenstions samples x trials.
	Returns an array of the same size where each trial is convolved with
	a gaussian kernel with sigma = sigma.
	"""
	##remove singleton dimesions and make sure values are floats
	array = array.squeeze().astype(float)
	##allocate memory for result
	result = np.zeros(array.shape)
	##if the array is 2-D, handle each trial separately
	try:
		for trial in range(array.shape[1]):
			result[:,trial] = gaussian_filter(array[:, trial], sigma = sigma, order = 0, mode = "constant", cval = 0.0)
	##if it's 1-D:
	except IndexError:
		if array.shape[0] == array.size:
			result = gaussian_filter(array, sigma = sigma, order = 0, mode = "constant", cval = 0.0)
		else:
			print "Check your array dimenszions!"
	return result

##a function to extract the creation date (expressed as the 
##julian date) in integer format of a given filepath
def get_cdate(path):
	return int(time.strftime("%j", time.localtime(os.path.getctime(path))))


##takes in a dictionary returned by parse_log and returns the 
##percent correct. Chance is the rewarded chance rate for the active lever.
##function assumes the best possible performance is the chance rate.
def get_p_correct(result_dict, chance = 0.9):
	total_trials = len(result_dict['top_lever'])+len(result_dict['bottom_lever'])
	correct_trials = len(result_dict['rewarded_poke'])
	return (float(correct_trials)/total_trials)/chance

##takes in a dictionary returned by parse_log and returns the 
##success rate (mean for the whole session)
def get_success_rate(result_dict):
	correct_trials = len(result_dict['rewarded_poke'])
	session_len = result_dict['session_length'][0]/60.0
	return float(correct_trials)/session_len

##returns a list of file paths for all log files in a directory
def get_log_file_names(directory):
	##get the current dir so you can return to it
	cd = os.getcwd()
	filepaths = []
	os.chdir(directory)
	for f in glob.glob("*.txt"):
		filepaths.append(os.path.join(directory,f))
	os.chdir(cd)
	return filepaths

