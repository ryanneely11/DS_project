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
	win_size=0.5,win_step=0.25,smooth=10):
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
	coeffs,sig_vals,epoch_idx,num_units = mr.regress_everything(f_behavior,f_ephys,
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
	fig.suptitle("Proportion significant of "+str(num_units)+" units",fontsize=14)



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
	win_size=0.5,win_step=0.25,smooth=10):
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
	coeffs,sig_vals,epoch_idx,num_units = mr.regress_session_epochs(f_behavior,f_ephys,
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
	fig.suptitle("Proportion sig. of "+str(num_units)+" units "+f_behavior[-11:-5],fontsize=14)


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



