###plotting.py:
##a series of functions for plotting data

import ml_regress as mr
import parse_timestamps as pt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ArrowStyle
import matplotlib.gridspec as gridspec

"""
A function to plot the significance of the regression
coefficients for a all units over epochs defined elsewhere, for all sessions* (see file_lists.py)
Exaclt the same as plot_session_regression but it includes data from multple sessions
Inputs:
	-f_behavior: file path of HDF5 file containing the behavioral timestamps
	-f_ephys: file path to the HDF5 file containing the ephys data
	-epoch_durations: a list of the desired durations (in sec) of each behavioral epoch. 
		the order is: choice (pre-action), peri-action (centered around the action ts),
		delay (time before the rewarde ts), and reward (time after the rewarde ts).
	-win_size = the size of each window  (in sec)
	-win_step = the time in s over which to slide the window over each epoch interval. 
		win_size = win_step means windows do not overlap. 
	-smooth: whether or not to use gaussian smoothing. Any value > 0 will smooth with
		a kernel of width smooth
"""
def plot_all_regressions(epoch_durations=[2,0.5,1,2],
	win_size=0.5,win_step=0.25,smooth=10,discard_non_sig=True):
	##assume that the epochs are the same from the ml_regress functions
	epoch_labels = ['Pre-action','Action','Delay','Outcome']
	epochs = ['choice','action','delay','reward']
	colors = ['g','r','k','c']
	##assume the following regressors; in this order
	regressors = ['Choice','Prv. Choice','Reward',
					'Prv. Reward','Choice val',
					'Rx time']
	num_windows = pt.get_num_windows(epoch_durations,win_size,win_step)
	##the x-axis 
	x_coords = np.linspace(0,sum(epoch_durations),num_windows)
	##get the data
	coeffs,sig_vals,epoch_idx,num_units = mr.regress_everything(epoch_durations=epoch_durations,
		win_size=win_size,win_step=win_step,smooth=smooth,save=False,discard_non_sig=discard_non_sig)
	##setup the figure to plot the significant values
	fig = plt.figure()
	gs = gridspec.GridSpec(len(regressors),num_windows)
	for r in range(len(regressors)):
		for e in range(len(epochs)):
			epoch = epochs[e]
			idx = epoch_idx[epoch]
			x = x_coords[idx]
			ax = plt.subplot(gs[r,idx[0]:idx[-1]+1])
			#ax.axhspan(0,0.05,facecolor='b',alpha=0.5) ##significance threshold
			if x.size == 1:
				x_center = x[0]/2.0
			else:
				x_center = (x[-1]-x[0])/2
			color = colors[e]
			ydata = sig_vals[r,idx]
			ax.plot(x,ydata,color=color,linewidth=2,marker='o',label=epoch)
			# for xpt,ypt in zip(x,ydata):
			# 	if ypt <=0.05:
			# 		ax.text(xpt,ypt+0.1,"*",fontsize=16) ##TODO: figure out how to get significance here
			ax.set_ylim(0,0.6)
			##conditional axes labels
			if r+1 == len(regressors):
				ax.set_xticks(np.round(x,1))
			else:
				ax.set_xticklabels([])
			if e+1 == len(epochs):
				ax.yaxis.tick_right()
				ax.set_yticks([0,0.3,0.6])
			else:
				ax.set_yticklabels([])
			if r == 0:
				ax.set_title(epoch_labels[e],fontsize=14,weight='bold')
			if e == 0:
				ax.set_ylabel(regressors[r],fontsize=12,weight='bold')
			if r+1 == len(regressors) and e == 2:
				ax.set_xlabel("Time in trial, s",fontsize=14,weight='bold')
	fig.suptitle("Proportion of "+str(num_units)+" units",fontsize=14)
	#setup the figure to plot the coefficients
	fig = plt.figure()
	gs = gridspec.GridSpec(len(regressors),num_windows)
	for r in range(len(regressors)):
		for e in range(len(epochs)):
			epoch = epochs[e]
			idx = epoch_idx[epoch]
			x = x_coords[idx]
			ax = plt.subplot(gs[r,idx[0]:idx[-1]+1])
			#ax.axhspan(0,0.05,facecolor='b',alpha=0.5) ##significance threshold
			if x.size == 1:
				x_center = x[0]/2.0
			else:
				x_center = (x[-1]-x[0])/2
			color = colors[e]
			ydata = coeffs[r,idx]
			ax.plot(x,ydata,color=color,linewidth=2,marker='o',label=epoch)
			# for xpt,ypt in zip(x,ydata):
			# 	if ypt <=0.05:
			# 		ax.text(xpt,ypt+0.1,"*",fontsize=16) ##TODO: figure out how to get significance here
			ax.set_ylim(-1,1)
			##conditional axes labels
			if r+1 == len(regressors):
				ax.set_xticks(np.round(x,1))
			else:
				ax.set_xticklabels([])
			if e+1 == len(epochs):
				ax.yaxis.tick_right()
				#ax.set_yticks([-.5,.5])
			else:
				ax.set_yticklabels([])
			if r == 0:
				ax.set_title(epoch_labels[e],fontsize=14,weight='bold')
			if e == 0:
				ax.set_ylabel(regressors[r],fontsize=12,weight='bold')
			if r+1 == len(regressors) and e == 2:
				ax.set_xlabel("Time in trial, s",fontsize=14,weight='bold')
	fig.suptitle("Mean regression coeffs.",fontsize=14)


"""
A function to plot the significance of the regression
coefficients for a all units over epochs defined elsewhere, for one session
Inputs:
	-f_behavior: file path of HDF5 file containing the behavioral timestamps
	-f_ephys: file path to the HDF5 file containing the ephys data
	-epoch_durations: a list of the desired durations (in sec) of each behavioral epoch. 
		the order is: choice (pre-action), peri-action (centered around the action ts),
		delay (time before the rewarde ts), and reward (time after the rewarde ts).
	-win_size = the size of each window  (in sec)
	-win_step = the time in s over which to slide the window over each epoch interval. 
		win_size = win_step means windows do not overlap. 
	-smooth: whether or not to use gaussian smoothing. Any value > 0 will smooth with
		a kernel of width smooth
"""
def plot_session_regression(f_behavior,f_ephys,epoch_durations=[2,0.5,1,2],
	win_size=0.5,win_step=0.25,smooth=10,discard_non_sig=True):
	##assume that the epochs are the same from the ml_regress functions
	epoch_labels = ['Pre-action','Action','Delay','Outcome']
	epochs = ['choice','action','delay','reward']
	colors = ['g','r','k','c']
	##assume the following regressors; in this order
	regressors = ['Choice','Prv. Choice','Reward',
					'Prv. Reward','Choice val',
					'Rx time']
	num_windows = pt.get_num_windows(epoch_durations,win_size,win_step)
	##the x-axis 
	x_coords = np.linspace(0,sum(epoch_durations),num_windows)
	##get the data
	coeffs,sig_vals,epoch_idx,num_units = mr.regress_session_epochs(f_behavior,f_ephys,
		epoch_durations=epoch_durations,win_size=win_size,win_step=win_step,proportions=True,
		discard_non_sig=discard_non_sig)
	##setup the figure
	fig = plt.figure()
	gs = gridspec.GridSpec(len(regressors),num_windows)
	for r in range(len(regressors)):
		for e in range(len(epochs)):
			epoch = epochs[e]
			idx = epoch_idx[epoch]
			x = x_coords[idx]
			ax = plt.subplot(gs[r,idx[0]:idx[-1]+1])
			#ax.axhspan(0,0.05,facecolor='b',alpha=0.5) ##significance threshold
			if x.size == 1:
				x_center = x[0]/2.0
			else:
				x_center = (x[-1]-x[0])/2
			color = colors[e]
			ydata = sig_vals[r,idx]
			ax.plot(x,ydata,color=color,linewidth=2,marker='o',label=epoch)
			# for xpt,ypt in zip(x,ydata):
			# 	if ypt <=0.05:
			# 		ax.text(xpt,ypt+0.1,"*",fontsize=16) ##TODO: figure out how to get significance here
			ax.set_ylim(0,0.6)
			##conditional axes labels
			if r+1 == len(regressors):
				ax.set_xticks(np.round(x,1))
			else:
				ax.set_xticklabels([])
			if e+1 == len(epochs):
				ax.yaxis.tick_right()
				ax.set_yticks([0,0.3,0.6])
			else:
				ax.set_yticklabels([])
			if r == 0:
				ax.set_title(epoch_labels[e],fontsize=14,weight='bold')
			if e == 0:
				ax.set_ylabel(regressors[r],fontsize=12,weight='bold')
			if r+1 == len(regressors) and e == 2:
				ax.set_xlabel("Time in trial, s",fontsize=14,weight='bold')
	fig.suptitle("Proportion sig. of "+str(num_units)+" units "+f_behavior[-11:-5],fontsize=14)
	##setup the figure for coefficient plotting
	fig = plt.figure()
	gs = gridspec.GridSpec(len(regressors),num_windows)
	for r in range(len(regressors)):
		for e in range(len(epochs)):
			epoch = epochs[e]
			idx = epoch_idx[epoch]
			x = x_coords[idx]
			ax = plt.subplot(gs[r,idx[0]:idx[-1]+1])
			#ax.axhspan(0,0.05,facecolor='b',alpha=0.5) ##significance threshold
			if x.size == 1:
				x_center = x[0]/2.0
			else:
				x_center = (x[-1]-x[0])/2
			color = colors[e]
			ydata = coeffs[r,idx]
			ax.plot(x,ydata,color=color,linewidth=2,marker='o',label=epoch)
			# for xpt,ypt in zip(x,ydata):
			# 	if ypt <=0.05:
			# 		ax.text(xpt,ypt+0.1,"*",fontsize=16) ##TODO: figure out how to get significance here
			#ax.set_ylim(-0.8,0.8)
			##conditional axes labels
			if r+1 == len(regressors):
				ax.set_xticks(np.round(x,1))
			else:
				ax.set_xticklabels([])
			if e+1 == len(epochs):
				ax.yaxis.tick_right()
				#ax.set_yticks([-0.8,0,0.8])
			else:
				ax.set_yticklabels([])
			if r == 0:
				ax.set_title(epoch_labels[e],fontsize=14,weight='bold')
			if e == 0:
				ax.set_ylabel(regressors[r],fontsize=12,weight='bold')
			if r+1 == len(regressors) and e == 2:
				ax.set_xlabel("Time in trial, s",fontsize=14,weight='bold')
	fig.suptitle("Coefficient values"+f_behavior[-11:-5],fontsize=14)


"""
A function to plot the significance of the regression
coefficients for a single unit over epochs defined elsewhere
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
"""
def plot_unit_regression(f_behavior,f_ephys,unit_name,
						epoch_durations=[2,0.5,1,2],win_size=0.5,win_step=0.25,smooth=10):
	##assume that the epochs are the same from the ml_regress functions
	epoch_labels = ['Pre-action','Action','Delay','Outcome']
	epochs = ['choice','action','delay','reward']
	colors = ['g','r','k','c']
	##assume the following regressors; in this order
	regressors = ['Choice','Reward','C x R',
					'Q upper','Q lower',
					'Q chosen']
	num_windows = pt.get_num_windows(epoch_durations,win_size,win_step)
	##the x-axis 
	x_coords = np.linspace(0,sum(epoch_durations),num_windows)
	##get the data
	coeffs,sig_vals,epoch_idx = mr.regress_unit_epochs(f_behavior,f_ephys,unit_name,
		epoch_durations=[2,0.5,1,2],win_size=0.5,win_step=0.25)
	##setup the figure
	fig = plt.figure()
	gs = gridspec.GridSpec(len(regressors),num_windows)
	for r in range(len(regressors)):
		for e in range(len(epochs)):
			epoch = epochs[e]
			idx = epoch_idx[epoch]
			x = x_coords[idx]
			ax = plt.subplot(gs[r,idx[0]:idx[-1]+1])
			ax.axhspan(0,0.05,facecolor='b',alpha=0.5) ##significance threshold
			if x.size == 1:
				x_center = x[0]/2.0
			else:
				x_center = (x[-1]-x[0])/2
			color = colors[e]
			ydata = sig_vals[r,idx]
			ax.plot(x,ydata,color=color,linewidth=2,marker='o',label=epoch)
			for xpt,ypt in zip(x,ydata):
				if ypt <=0.05:
					ax.text(xpt,ypt+0.1,"*",fontsize=16)
			ax.set_ylim(0,1)
			##conditional axes labels
			if r+1 == len(regressors):
				ax.set_xticks(np.round(x,1))
			else:
				ax.set_xticklabels([])
			if e+1 == len(epochs):
				ax.yaxis.tick_right()
				ax.set_yticks([0,0.5,1])
			else:
				ax.set_yticklabels([])
			if r == 0:
				ax.set_title(epoch_labels[e],fontsize=14,weight='bold')
			if e == 0:
				ax.set_ylabel(regressors[r],fontsize=12,weight='bold')
			if r+1 == len(regressors) and e == 2:
				ax.set_xlabel("Time in trial, s",fontsize=14,weight='bold')
	fig.suptitle("P-val of regression coefficients, "+unit_name,fontsize=16)


##this function takes an hdf5 file and plots the trials arranged by duration
##in a schamatic way. 
def plot_trials(f_in):
	#get the relevant data
	results_dict = pt.sort_by_trial(f_in,save_data=False)
	colorkey=('g','b','r')
	##start with the upper lever rewarded sessions
	try:
		upper_trials = results_dict['upper_rewarded']
		upper_trial_durs = pt.get_trial_durations(upper_trials)
		#sort according to total trial len
		upper_idx = np.argsort(abs(upper_trial_durs))[::-1]
		upper_sorted = upper_trials[upper_idx]
		##plot 'em
		fig, ax = plt.subplots(1)
		for i in range(upper_sorted.shape[0]):
			##zero out the start time
			xdata = abs(upper_sorted[i,:])
			xdata = xdata-xdata[0]
			ydata = np.zeros(xdata.size)+(i+1)
			ax.plot(xdata,ydata,linewidth=1,color='k',alpha=0.5)
			ax.scatter(xdata,ydata,facecolors=colorkey,s=30)
		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(14)
		for tick in ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(14)
		ax.set_xlabel("Time from trial start (s)",fontsize=16)
		ax.set_ylabel("Trial number",fontsize=16)
		ax.set_title("Upper lever rewarded trials",fontsize=16)
	except KeyError:
		upper_trial_durs = None
	##move on to the lower lever if applicable
	try:
		lower_trials = results_dict['lower_rewarded']
		lower_trial_durs = pt.get_trial_durations(lower_trials)
		#sort according to total trial len
		lower_idx = np.argsort(abs(lower_trial_durs))[::-1]
		lower_sorted = lower_trials[lower_idx]
		##plot 'em
		fig2, ax2 = plt.subplots(1)
		for i in range(lower_sorted.shape[0]):
			##zero out the start time
			xdata = abs(lower_sorted[i,:])
			xdata = xdata-xdata[0]
			ydata = np.zeros(xdata.size)+(i+1)
			ax2.plot(xdata,ydata,linewidth=1,color='k',alpha=0.5)
			ax2.scatter(xdata,ydata,facecolors=colorkey,s=30)
		for tick in ax2.xaxis.get_major_ticks():
			tick.label.set_fontsize(14)
		for tick in ax2.yaxis.get_major_ticks():
			tick.label.set_fontsize(14)
		ax2.set_xlabel("Time from trial start (s)",fontsize=16)
		ax2.set_ylabel("Trial number",fontsize=16)
		ax2.set_title("Lower lever rewarded trials",fontsize=16)
	except KeyError:
		lower_trial_durs = None


###this function plots the performance ACROSS DAYS of a given animal.
##It takes in a directiry where the raw (.txt) log files are stored.
##NOTE THAT THIS FUNCTION OPERATES ON RAW .txt FILES!!!!
##this allows for an accurate plotting based in the date recorded
def plot_epoch(directory,plot=True):
	##grab a list of all the logs in the given directory
	fnames = pt.get_log_file_names(directory)
	##x-values are the julian date of the session
	dates = [pt.get_cdate(f) for f in fnames]
	##y-values are the success rates (or percent correct?) for each session
	scores = []
	for session in fnames:
		# print "working on session "+ session
		result = pt.parse_log(session)
		# print "score is "+str(get_success_rate(result))
		scores.append(pt.get_success_rate(result))
	##convert lists to arrays for the next steps
	dates = np.asarray(dates)
	scores = np.asarray(scores)
	##files may not have been opened in order of ascending date, so sort them
	sorted_idx = np.argsort(dates)
	dates = dates[sorted_idx]
	##adjust dates so they start at 0
	dates = dates-(dates[0]-1)
	scores = scores[sorted_idx]
	##we want to not draw lines when there are non-consecutive training days:
	##our x-axis will then be a contiuous range of days
	x = range(1,dates[-1]+1)
	##insert None values in the score list when a date was skipped
	skipped = []
	for idx, date in enumerate(x):
		if date not in dates:
			scores = np.insert(scores,idx,np.nan)
	if plot:
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(x, scores, 'o', color = "c")
		ax.set_xlabel("Training day")
		ax.set_ylabel("Correct trials per min")
	return x, dates, scores


"""
takes in a data dictionary produced by parse_log
plots the lever presses and the switch points for levers
"""
def plot_presses(f_in, sigma = 20):
	##extract relevant data
	data_dict = h5py.File(f_in,'r')
	top = data_dict['top_lever']
	bottom = data_dict['bottom_lever']
	duration = int(np.ceil(data_dict['session_length']))
	top_rewarded = np.asarray(data_dict['top_rewarded'])/60.0
	bottom_rewarded = np.asarray(data_dict['bottom_rewarded'])/60.0
	##convert timestamps to histogram structures
	top, edges = np.histogram(top, bins = duration)
	bottom, edges = np.histogram(bottom, bins = duration)
	##smooth with a gaussian window
	top = pt.gauss_convolve(top, sigma)
	bottom = pt.gauss_convolve(bottom, sigma)
	##get plotting stuff
	data_dict.close()
	x = np.linspace(0,np.ceil(duration/60.0), top.size)
	mx = max(top.max(), bottom.max())
	mn = min(top.min(), bottom.min())
	fig = plt.figure()
	gs = gridspec.GridSpec(2,2)
	ax = fig.add_subplot(gs[0,:])
	ax2 = fig.add_subplot(gs[1,0])
	ax3 = fig.add_subplot(gs[1,1], sharey=ax2)
	##the switch points
	ax.vlines(top_rewarded, mn, mx, colors = 'r', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "top rewarded")
	ax.vlines(bottom_rewarded, mn, mx, colors = 'b', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "bottom rewarded")
	ax2.vlines(top_rewarded, mn, mx, colors = 'r', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "top rewarded")
	ax2.vlines(bottom_rewarded, mn, mx, colors = 'b', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "bottom rewarded")
	ax3.vlines(top_rewarded, mn, mx, colors = 'r', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "top rewarded")
	ax3.vlines(bottom_rewarded, mn, mx, colors = 'b', linestyles = 'dashed', 
		linewidth = '2', alpha = 0.5, label = "bottom rewarded")
	ax.plot(x, top, color = 'r', linewidth = 2, label = "top lever")
	ax.plot(x, bottom, color = 'b', linewidth = 2, label = "bottom_lever")
	ax.legend()
	ax.set_ylabel("press rate", fontsize = 14)
	fig.suptitle("Lever press performance", fontsize = 18)
	ax.set_xlim(-1, x[-1]+1)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	#plot them separately
	##figure out the order of lever setting to create color spans
	# if top_rewarded.min() < bottom_rewarded.min():
	# 	for i in range(top_rewarded.size):
	# 		try:
	# 			ax2.axvspan(top_rewarded[i], bottom_rewarded[i], facecolor = 'r', alpha = 0.2)
	# 		except IndexError:
	# 			ax2.axvspan(top_rewarded[i], duration, facecolor = 'r', alpha = 0.2)
	# else:
	# 	for i in range(bottom_rewarded.size):
	# 		try:
	# 			ax3.axvspan(bottom_rewarded[i], top_rewarded[i], facecolor = 'b', alpha = 0.2)
	# 		except IndexError:
	# 			ax3.axvspan(bottom_rewarded[i], duration, facecolor = 'b', alpha = 0.2)
	ax2.plot(x, top, color = 'r', linewidth = 2, label = "top lever")
	ax3.plot(x, bottom, color = 'b', linewidth = 2, label = "bottom_lever")
	ax2.set_ylabel("press rate", fontsize = 14)
	ax2.set_xlabel("Time in session, mins", fontsize = 14)
	ax3.set_xlabel("Time in session, mins", fontsize = 14)
	fig.suptitle("Lever press performance", fontsize = 18)
	ax2.set_xlim(-1, x[-1]+1)
	ax3.set_xlim(-1, x[-1]+1)
	for tick in ax2.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax3.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax2.set_title("top only", fontsize = 14)
	ax3.set_title("bottom only", fontsize = 14)



"""
Caclulates and plots trial duration for a single session.
Inputs:
	-dictionary of results produced by parse_timestamps
Outputs:
	dictionary of various ITI statistics
	also plots, if desired
"""
def trial_duration_analysis(f_in):
	##get the relevant data
	results_dict = pt.sort_by_trial(f_in,save_data=False)
	##start with the upper lever rewarded sessions
	try:
		upper_trials = results_dict['upper_rewarded']
		upper_trial_durs = pt.get_trial_durations(upper_trials)
	except KeyError:
		upper_trial_durs = None
	##move on to the lower lever if applicable
	try:
		lower_trials = results_dict['lower_rewarded']
		lower_trial_durs = pt.get_trial_durations(lower_trials)
	except KeyError:
		lower_trial_durs = None
	##get some basic stats
	fig,(ax,ax2) = plt.subplots(2,1,sharex=True)
	fig.patch.set_facecolor('white')
	fig.set_size_inches(10,4)
	if upper_trial_durs is not None:
		upper_dur_mean = abs(upper_trial_durs).mean()
		upper_dur_std = abs(upper_trial_durs).std()
		##outliers more than 2 std dev
		upper_outliers = abs(upper_trial_durs[np.where(abs(upper_trial_durs)>3*upper_dur_std)])
		##get just the successful trials
		r_idx = np.where(upper_trial_durs>0)
		r_upper_durs = upper_trial_durs[r_idx]
		r_upper_times = upper_trials[r_idx,0]
		##get just the unsuccessful trials
		u_idx = np.where(upper_trial_durs<0)
		u_upper_durs = upper_trial_durs[u_idx]
		u_upper_times = upper_trials[u_idx,0]
		##plot this stuff
		ax.scatter(r_upper_times,abs(r_upper_durs),edgecolor='green',marker='o',s=30,
			linewidth=2,facecolors=('green',),alpha=0.7,label='rewarded upper lever')
		ax.scatter(u_upper_times,abs(u_upper_durs),color='green',marker='x',s=30,
			linewidth=2,label='unrewarded upper lever')
		ax2.scatter(r_upper_times,abs(r_upper_durs),edgecolor='green',marker='o',s=30,
			linewidth=2,facecolors=('green',),alpha=0.7,label='rewarded upper lever')
		ax2.scatter(u_upper_times,abs(u_upper_durs),color='green',marker='x',s=30,
			linewidth=2,label='unrewarded upper lever')		
	if lower_trial_durs is not None:
		lower_dur_mean = abs(lower_trial_durs).mean()
		lower_dur_std = abs(lower_trial_durs).std()
		##outliers
		lower_outliers = abs(lower_trial_durs[np.where(abs(lower_trial_durs)>3*lower_dur_std)])
		##get just the successful trials
		r_idx = np.where(lower_trial_durs>0)
		r_lower_durs = lower_trial_durs[r_idx]
		r_lower_times = lower_trials[r_idx,0]
		##get just the unsuccessful trials
		u_idx = np.where(lower_trial_durs<0)
		u_lower_durs = lower_trial_durs[u_idx]
		u_lower_times = lower_trials[u_idx,0]
		##plot this stuff
		ax.scatter(r_lower_times,abs(r_lower_durs),edgecolor='red',marker='o',s=30,
			linewidth=2,facecolors=('red',),alpha=0.7,label='rewarded lower lever')
		ax.scatter(u_lower_times,abs(u_lower_durs),color='red',marker='x',s=30,
			linewidth=2,label='unrewarded lower lever')
		ax2.scatter(r_lower_times,abs(r_lower_durs),edgecolor='red',marker='o',s=30,
			linewidth=2,facecolors=('red',),alpha=0.7,label='rewarded lower lever')
		ax2.scatter(u_lower_times,abs(u_lower_durs),color='red',marker='x',s=30,
			linewidth=2,label='unrewarded lower lever')
	for label in ax2.xaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax.xaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax2.xaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	for label in ax.xaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	for label in ax2.yaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax.yaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax2.yaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	for label in ax.yaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	##if there are outliers, break the axis
	try:
		outliers = np.hstack((upper_outliers,lower_outliers))
	except NameError: ##if we only have one kind of trial
		if upper_trial_durs is not None:
			outliers = upper_outliers
		else:
			outliers = lower_outliers
	if outliers.size > 0:
		ax2.set_ylim(-1,max(2*lower_dur_std,2*upper_dur_std))
		ax.set_ylim(outliers.min()-5,outliers.max()+10)
		# hide the spines between ax and ax2
		ax.spines['bottom'].set_visible(False)
		ax2.spines['top'].set_visible(False)
		ax.xaxis.tick_top()
		ax.tick_params(labeltop='off')  # don't put tick labels at the top
		ax2.xaxis.tick_bottom()

		# This looks pretty good, and was fairly painless, but you can get that
		# cut-out diagonal lines look with just a bit more work. The important
		# thing to know here is that in axes coordinates, which are always
		# between 0-1, spine endpoints are at these locations (0,0), (0,1),
		# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
		# appropriate corners of each of our axes, and so long as we use the
		# right transform and disable clipping.

		d = .015  # how big to make the diagonal lines in axes coordinates
		# arguments to pass plot, just so we don't keep repeating them
		kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
		ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
		ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

		kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
		ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
		ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
	else:
		fig.delaxes(ax)
		fig.draw()
	if outliers.size > 0:
		legend=ax.legend(frameon=False)
		try:
			plt.text(0.2, 0.9,'upper lever mean = '+str(upper_dur_mean),ha='center',
				va='center',transform=ax.transAxes,fontsize=14)
		except NameError:
			pass
		try:
			plt.text(0.2, 0.8,'lower lever mean = '+str(lower_dur_mean),ha='center',
				va='center',transform=ax.transAxes,fontsize=14)
		except NameError:
			pass
	else:
		legend=ax2.legend(frameon=False)
	for label in legend.get_texts():
		label.set_fontsize('large')
	legend.get_frame().set_facecolor('none')
	ax2.set_xlabel("Time in session, s",fontsize=14)
	ax2.set_ylabel("Trial duration, s",fontsize=14)
	fig.suptitle("Duration of trials",fontsize=14)


"""
Caclulates and plots the interval between action and outcome
 for a single session.
Inputs:
	-dictionary of results produced by parse_timestamps
Outputs:
	plot
"""
def ao_duration_analysis(f_in):
	##get the relevant data
	results_dict = pt.sort_by_trial(f_in,save_data=False)
	##start with the upper lever rewarded sessions
	try:
		upper_trials = results_dict['upper_rewarded']
		upper_trial_durs = pt.get_ao_interval(upper_trials)
	except KeyError:
		upper_trial_durs = None
	##move on to the lower lever if applicable
	try:
		lower_trials = results_dict['lower_rewarded']
		lower_trial_durs = pt.get_ao_interval(lower_trials)
	except KeyError:
		lower_trial_durs = None
	##get some basic stats
	fig,(ax,ax2) = plt.subplots(2,1,sharex=True)
	fig.patch.set_facecolor('white')
	fig.set_size_inches(10,4)
	if upper_trial_durs is not None:
		upper_dur_mean = abs(upper_trial_durs).mean()
		upper_dur_std = abs(upper_trial_durs).std()
		##outliers more than 2 std dev
		upper_outliers = abs(upper_trial_durs[np.where(abs(upper_trial_durs)>3*upper_dur_std)])
		##get just the successful trials
		r_idx = np.where(upper_trial_durs>0)
		r_upper_durs = upper_trial_durs[r_idx]
		r_upper_times = upper_trials[r_idx,0]
		##get just the unsuccessful trials
		u_idx = np.where(upper_trial_durs<0)
		u_upper_durs = upper_trial_durs[u_idx]
		u_upper_times = upper_trials[u_idx,0]
		##plot this stuff
		ax.scatter(r_upper_times,abs(r_upper_durs),edgecolor='green',marker='o',s=30,
			linewidth=2,facecolors=('green',),alpha=0.7,label='rewarded upper lever')
		ax.scatter(u_upper_times,abs(u_upper_durs),color='green',marker='x',s=30,
			linewidth=2,label='unrewarded upper lever')
		ax2.scatter(r_upper_times,abs(r_upper_durs),edgecolor='green',marker='o',s=30,
			linewidth=2,facecolors=('green',),alpha=0.7,label='rewarded upper lever')
		ax2.scatter(u_upper_times,abs(u_upper_durs),color='green',marker='x',s=30,
			linewidth=2,label='unrewarded upper lever')		
	if lower_trial_durs is not None:
		lower_dur_mean = abs(lower_trial_durs).mean()
		lower_dur_std = abs(lower_trial_durs).std()
		##outliers
		lower_outliers = abs(lower_trial_durs[np.where(abs(lower_trial_durs)>3*lower_dur_std)])
		##get just the successful trials
		r_idx = np.where(lower_trial_durs>0)
		r_lower_durs = lower_trial_durs[r_idx]
		r_lower_times = lower_trials[r_idx,0]
		##get just the unsuccessful trials
		u_idx = np.where(lower_trial_durs<0)
		u_lower_durs = lower_trial_durs[u_idx]
		u_lower_times = lower_trials[u_idx,0]
		##plot this stuff
		ax.scatter(r_lower_times,abs(r_lower_durs),edgecolor='red',marker='o',s=30,
			linewidth=2,facecolors=('red',),alpha=0.7,label='rewarded lower lever')
		ax.scatter(u_lower_times,abs(u_lower_durs),color='red',marker='x',s=30,
			linewidth=2,label='unrewarded lower lever')
		ax2.scatter(r_lower_times,abs(r_lower_durs),edgecolor='red',marker='o',s=30,
			linewidth=2,facecolors=('red',),alpha=0.7,label='rewarded lower lever')
		ax2.scatter(u_lower_times,abs(u_lower_durs),color='red',marker='x',s=30,
			linewidth=2,label='unrewarded lower lever')
	for label in ax2.xaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax.xaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax2.xaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	for label in ax.xaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	for label in ax2.yaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax.yaxis.get_ticklabels()[1::2]:
		label.set_visible(False)
	for label in ax2.yaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	for label in ax.yaxis.get_ticklabels()[::2]:
		label.set_fontsize(14)
	##if there are outliers, break the axis
	try:
		outliers = np.hstack((upper_outliers,lower_outliers))
	except NameError: ##if we only have one kind of trial
		if upper_trial_durs is not None:
			outliers = upper_outliers
		else:
			outliers = lower_outliers
	if outliers.size > 0:
		ax2.set_ylim(-1,max(2*lower_dur_std,2*upper_dur_std))
		ax.set_ylim(outliers.min()-5,outliers.max()+10)
		# hide the spines between ax and ax2
		ax.spines['bottom'].set_visible(False)
		ax2.spines['top'].set_visible(False)
		ax.xaxis.tick_top()
		ax.tick_params(labeltop='off')  # don't put tick labels at the top
		ax2.xaxis.tick_bottom()

		# This looks pretty good, and was fairly painless, but you can get that
		# cut-out diagonal lines look with just a bit more work. The important
		# thing to know here is that in axes coordinates, which are always
		# between 0-1, spine endpoints are at these locations (0,0), (0,1),
		# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
		# appropriate corners of each of our axes, and so long as we use the
		# right transform and disable clipping.

		d = .015  # how big to make the diagonal lines in axes coordinates
		# arguments to pass plot, just so we don't keep repeating them
		kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
		ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
		ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

		kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
		ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
		ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
	else:
		fig.delaxes(ax)
		fig.draw()
	if outliers.size > 0:
		legend=ax.legend(frameon=False)
		try:
			plt.text(0.2, 0.9,'upper lever mean = '+str(upper_dur_mean),ha='center',
				va='center',transform=ax.transAxes,fontsize=14)
		except NameError:
			pass
		try:
			plt.text(0.2, 0.8,'lower lever mean = '+str(lower_dur_mean),ha='center',
				va='center',transform=ax.transAxes,fontsize=14)
		except NameError:
			pass
	else:
		legend=ax2.legend(frameon=False)
	for label in legend.get_texts():
		label.set_fontsize('large')
	legend.get_frame().set_facecolor('none')
	ax2.set_xlabel("Time in session, s",fontsize=14)
	ax2.set_ylabel("Trial duration, s",fontsize=14)
	fig.suptitle("Duration of trials",fontsize=14)

###****MULTI-SESSION ANALYSIS*****####

##this function takes in a list of directories where RAW(!) .txt logs are stored
def plot_epochs_multi(directories):
	
	##assume the folder name is the animal name, and that is is two chars
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for d in directories:
		name = d[-11:-9]
		x, dates, scores = plot_epoch(d, plot = False)
		##get a random color to plot this data with
		c = np.random.rand(3,)
		ax.plot(x, scores, 's', markersize = 10, color = c)
		ax.plot(x, scores, linewidth = 2, color = c, label = name)
	ax.legend(loc=2)
	##add horizontal lines showing surgery and recording days
	x_surg = [16,17,18,19]
	y_surg = [-.1,-.1,-.1,-.1]
	x_rec = range(25,44)
	y_rec = np.ones(len(x_rec))*-0.1
	x_pre = range(0,15)
	y_pre = np.ones(len(x_pre))*-0.1
	ax.plot(x_surg, y_surg, linewidth = 4, color = 'k')
	ax.plot(x_pre, y_pre, linewidth = 4, color = 'c')
	ax.plot(x_rec, y_rec, linewidth =4, color = 'r')
	ax.text(15,-0.32, "Surgeries", fontsize = 16)
	ax.text(32,-0.32, "Recording", fontsize = 16)
	ax.text(4,-0.32, "Pre-training", fontsize = 16)		
	ax.set_xlabel("Training day", fontsize=16)
	ax.set_ylabel("Correct trials per min", fontsize=16)
	fig.suptitle("Performance across days", fontsize=16)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)