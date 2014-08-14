#!/usr/bin/env python
# encoding: utf-8
"""
BehaviorOperator.py

Created by Tomas Knapen on 2010-11-06.
Modified by Dirk van Moorselaar on 2014-05-23
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess
import tempfile, logging, pickle

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

from nifti import *
from pypsignifit import *
from Tools.Operators.BehaviorOperator import NewBehaviorOperator, TrialEventSequence
from IPython import embed as shell
import re


class WMMBehaviorOperator(NewBehaviorOperator):
	def __init__(self, inputObject, **kwargs):
		"""docstring for __init__"""
		super(WMMBehaviorOperator, self).__init__(inputObject = inputObject, **kwargs)
		with open( self.inputFileName ) as f:
			file_data = pickle.load(f)
		self.events = file_data['eventArray']
		self.parameters = file_data['parameterArray']
		
		run_start_time_string = [e for e in self.events[0] if e[:len('trial 0 phase 1')] == 'trial 0 phase 1']
		self.run_start_time = float(run_start_time_string[0].split(' ')[-1])
					
	def phase_timings(self):
		"""
		Function that returns array with length of nr_of_trials that contains timings (corrected for run_start_time) of all phases of the experiment. 

		First with recompile, the raw timings of each phase are extracted from the event parameters.
		Next all phase_events are stored in separate lists for all trials 
		"""
		self.phase_events = []
		for j in range (len(self.events)):
			rec_phase = re.compile('trial %d phase (\d+) started at (-?\d+\.?\d*)' % j)
			self.phase_events.append(filter(None,[re.findall(rec_phase,self.events[j][i]) for i in range (len(self.events[j])) if isinstance (self.events[j][i],str)]))
		
		for a in range(len(self.phase_events)):
			for b in range(len(self.phase_events[a])):
				self.phase_events[a][b] = [self.phase_events[a][b][0][0], float(self.phase_events[a][b][0][1]) - self.run_start_time]
		return self.phase_events	
		
	def response_timings(self):
		"""
		Function that returns array with length of nr_of_trials that contains timings (corrected for run_start_time) of all phases of the experiment. 
		Function is similar to phase timings. Note that in contrast to phase_timings, function can return lists of different lengths depending
		on whether or not a pp responded more than once on a specific trial 
		
		"""
		self.response_events = []
		for j in range (len(self.events)):	
			rec_button = re.compile('trial %d event ([b,y]) at (-?\d+\.?\d*)' % j)
			self.response_events.append(filter(None,[re.findall(rec_button,self.events[j][i]) for i in range (len(self.events[j])) if isinstance (self.events[j][i],str)]))    
            
		for a in range(len(self.response_events)):
			for b in range(len(self.response_events[a])):
				self.response_events[a][b] = [self.response_events[a][b][0][0], float(self.response_events[a][b][0][1]) - self.run_start_time]
		
		return self.response_events
			
	def trial_info (self, keys = ['answer']):
		"""
		function that returns list of arrays with trial information. Per trial all information in keys will be returned.
		"""	
		
		self.trial_info = [[self.parameters[i][key] for key in keys] for i in range(len(self.parameters))]
		
		return self.trial_info
		
		
