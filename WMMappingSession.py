#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Modified by Dirk van Moorselaar on 2014-05-23
Copyright (c) 2009 TK. All rights reserved.
"""
# import core functionality:
import os, sys, subprocess, datetime
import tempfile, logging, pickle
import numpy as np
import scipy as sp
from scipy.stats import *
from scipy.stats import norm
import matplotlib.pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
import random
from random import *
import bottleneck as bn
from itertools import *
from IPython import embed as shell

# import functionality:
import mne
import nitime
#from skimage import *
import sklearn
from nifti import *
from pypsignifit import *
from nitime import fmri
from scipy import misc
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.stats.stats import pearsonr
from skimage import filter
#from skimage.transform import rescale, resize
#import matplotlib.pyplot as plt


# import custom functionality:
# from Tools.Sessions import Session 
# from Tools.Operators.PhysioOperator import PhysioOperator
# from Tools.Operators.CommandLineOperator import FEATOperator
from Tools.Sessions import *
from Tools.Run import *
from Tools.Operators import *
from WMMBehaviorOperator import *
from Tools.Operators.PhysioOperator import PhysioOperator

class WMMappingSession(Session):
	"""
	Template Class for fMRI sessions analysis.
	"""
	def __init__(self, ID, date, project, subject, parallelize = True, loggingLevel = logging.DEBUG):
		super(WMMappingSession, self).__init__(ID, date, project, subject, parallelize = parallelize, loggingLevel = loggingLevel)
	

	###########################################################################################################################################
	######															Timing analysis:													 ######
	######	Functions that analyze all relevant timing parameters (behavioral and eyemovement data)						 				 ######		
	######																													 			 ######	
	###########################################################################################################################################	

	def stimulus_response_timings(self):
		"""stimulus_timings uses behavior operators to distil:
		- the times at which stimulus presentation began and ended per task type
		- the times at which the task buttons were pressed. 
		- stores text file for glm and for decoding purposes (text file with patch info)
		"""
		
		for run in self.conditionDict['WMM']:
			# per run get all stimulus presentations and responnse data
			bO = WMMBehaviorOperator(self.runFile(stage = 'processed/behavior', run = self.runList[run], extension = '.dat' ))
			phase_timing = bO.phase_timings() # sets up timing parameters
			response_timing = bO.response_timings()
			trial_info = bO.trial_info(keys = ['answer','answer_time','rotation_patch','rotation_cross','spatial_patch','start_cross'])

			# store text file that contains all stimulus presentations per run (only memory presentations) for glm
			all_timings = np.array([[phase_timing[t][1][1],phase_timing[t][2][1]-phase_timing[t][1][1],1.0]for t in range(len(trial_info))])

			# and for decoding purposes
			all_timings_task = np.array([[all_timings[t][0],phase_timing[t][3][1]]for t in range(len(trial_info))])
			all_timings_task = np.concatenate((all_timings_task, np.array(trial_info)[:,4]. reshape(len(np.array(trial_info)[:,4]),1)),axis = 1)
			
			if run % 2 == 0:
				all_timings_task = np.concatenate((all_timings_task,np.vstack((np.ones((20,1)),np.zeros((20,1))))),axis = 1)
			elif run % 2 == 1:
				all_timings_task = np.concatenate((all_timings_task,np.vstack((np.zeros((20,1)),np.ones((20,1))))),axis = 1)

			np.savetxt(self.runFile(stage = 'processed/behavior', run = self.runList[run], extension = '.txt', postFix = ['stim' ,'all']), all_timings, fmt = '%3.2f', delimiter = '\t')
			np.savetxt(self.runFile(stage = 'processed/behavior', run = self.runList[run], extension = '.txt', postFix = ['stim' ,'all','task']), all_timings_task, fmt = '%3.2f', delimiter = '\t')

			# store text file that contains all response timings per run (only memory presentations)
			all_timings = []
			for i in range(len(response_timing)):
				for j in range(len(response_timing[i])):
					all_timings.append([response_timing[i][j][1],0.50,1.0])
	
			np.savetxt(self.runFile(stage = 'processed/behavior', run = self.runList[run], extension = '.txt', postFix = ['resp' ,'all']), np.array(all_timings), fmt = '%3.2f', delimiter = '\t')

	def stimulus_timings_unique_PRF(self):
		"""stimulus_timings uses behavior operators to distil:
		- the times at which stimulus presentation began and ended per task type ()
		- the times at which the task buttons were pressed. 
		"""
		
		for run in self.conditionDict['WMM']:
			# per run get all stimulus presentations and responnse data
			bO = WMMBehaviorOperator(self.runFile(stage = 'processed/behavior', run = self.runList[run], extension = '.dat' ))
			phase_timing = bO.phase_timings() # sets up timing parameters
			response_timing = bO.response_timings()
			trial_info = bO.trial_info(keys = ['answer','answer_time','rotation_patch','rotation_cross','spatial_patch','start_cross'])
			
			patches_task_1 = [trial_info[i][4] for i in range(len(trial_info)/2)]
			patches_task_2 = [trial_info[i][4] for i in range(len(trial_info)/2,len(trial_info))]

			unique_task_1 = []
			unique_task_2 = []
			index_task_1 = []
			index_task_2 = []
			
			for i in range(len(patches_task_1)):
				if patches_task_1[i] not in unique_task_1:
					unique_task_1.append(patches_task_1[i])
					index_task_1.append(i)
				if patches_task_2[i] not in unique_task_2:
					unique_task_2.append(patches_task_2[i])
					index_task_2.append(i + len(trial_info)/2)	

			timings_task_1_start = []
			timings_task_2_start = []
			timings_task_1_end = []
			timings_task_2_end = []

			for i in range(len(index_task_1)):
				timings_task_1_start.append([phase_timing[index_task_1[i]][1][1],phase_timing[i][2][1]-phase_timing[i][1][1],unique_task_1[i]])
				timings_task_2_start.append([phase_timing[index_task_2[i]][1][1],phase_timing[i][2][1]-phase_timing[i][1][1],unique_task_2[i]])
				timings_task_1_end.append([phase_timing[index_task_1[i]][3][1],phase_timing[i][4][1]-phase_timing[i][3][1],unique_task_1[i]])
				timings_task_2_end.append([phase_timing[index_task_2[i]][3][1],phase_timing[i][4][1]-phase_timing[i][3][1],unique_task_2[i]])				

			if run % 2 == 0:
				task_order = ['patch','center']
			elif run % 2 == 1:
				task_order = ['center','patch']
			
			np.savetxt(self.runFile(stage = 'processed/behavior', run = self.runList[run], extension = '.txt', postFix = ['unique' ,task_order[0],'start']), np.array(timings_task_1_start), fmt = '%3.2f', delimiter = '\t')	
			np.savetxt(self.runFile(stage = 'processed/behavior', run = self.runList[run], extension = '.txt', postFix = ['unique' ,task_order[1],'start']), np.array(timings_task_2_start), fmt = '%3.2f', delimiter = '\t')
			np.savetxt(self.runFile(stage = 'processed/behavior', run = self.runList[run], extension = '.txt', postFix = ['unique' ,task_order[0],'end']), np.array(timings_task_1_end), fmt = '%3.2f', delimiter = '\t')	
			np.savetxt(self.runFile(stage = 'processed/behavior', run = self.runList[run], extension = '.txt', postFix = ['unique' ,task_order[1],'end']), np.array(timings_task_2_end), fmt = '%3.2f', delimiter = '\t')

	def timings_across_runs(self):
		"""
		Function that creates one single text file per participant with relevant info for PRF analysis
		"""

		# first determine individual run duration (to make sure that stimulus timings of all runs are correct)
		run_duration = []
		for r in [self.runList[i] for i in self.conditionDict['WMM']]:
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = r))
			tr, nr_trs = round(niiFile.rtime*1)/1000.0, niiFile.timepoints
			run_duration.append(tr * nr_trs)
		run_duration = np.r_[0,np.cumsum(np.array(run_duration))]

		# timing information stimuli
		stim_info = []
		run = 0
		for r in [self.runList[i] for i in self.conditionDict['WMM']]:
			stim_events = np.loadtxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = ['stim' ,'all','task']))
			stim_events[:,:2] += run_duration[run]
			stim_info.append(stim_events)
			run += 1

		# save stim_info as text_file	
		np.savetxt(self.runFile(stage = 'processed/behavior', postFix = ['stim_info_all'],extension = '.txt'), np.vstack(stim_info), fmt = '%3.2f', delimiter = '\t')	
			
	def eye_timings(self, nr_dummy_scans = 6, mystery_threshold = 0.05,saccade_duration_threshold = 10):
		"""
		- the times at which a blink began per run 
		- duration of blink
		Timings of the blinks are corrected for the start of the scan by the nr_dummy_scans
		"""

	
		for r in [self.runList[i] for i in self.conditionDict['WMM']]:
			# shell()
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = r))
			tr = round(niiFile.rtime*1)/1000.0
			with open (self.runFile(stage = 'processed/eye', run = r, extension = '.msg')) as inputFileHandle:
				msg_file = inputFileHandle.read()


			sacc_re = 'ESACC\t(\S+)[\s\t]+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+.?\d+)'
			fix_re = 'EFIX\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?'
			blink_re = 'EBLINK\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d?.?\d*)?'
			start_eye = 'START\t(-?\d+\.?\d*)'

			# self.logger.info('reading eyelink events from %s', os.path.split(self.message_file)[-1])
			saccade_strings = re.findall(re.compile(sacc_re), msg_file)
			fix_strings = re.findall(re.compile(fix_re), msg_file)
			blink_strings = re.findall(re.compile(blink_re), msg_file)
			start_time_scan = float(re.findall(re.compile(start_eye),msg_file)[0])
			
			if len(saccade_strings) > 0:
				self.saccades_from_message_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3]),'start_x':float(e[4]),'start_y':float(e[5]),'end_x':float(e[6]),'end_y':float(e[7]), 'mystery_measure':float(e[8]),'peak_velocity':float(e[9])} for e in saccade_strings]
				self.fixations_from_message_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3]),'x':float(e[4]),'y':float(e[5]),'pupil_size':float(e[6])} for e in fix_strings]
				self.blinks_from_message_file = [{'eye':e[0],'start_timestamp':float(e[1]),'end_timestamp':float(e[2]),'duration':float(e[3])} for e in blink_strings]
			
				self.saccade_type_dictionary = np.dtype([(s , np.array(self.saccades_from_message_file[0][s]).dtype) for s in self.saccades_from_message_file[0].keys()])
				self.fixation_type_dictionary = np.dtype([(s , np.array(self.fixations_from_message_file[0][s]).dtype) for s in self.fixations_from_message_file[0].keys()])
				if len(self.blinks_from_message_file) > 0:
					self.blink_type_dictionary = np.dtype([(s , np.array(self.blinks_from_message_file[0][s]).dtype) for s in self.blinks_from_message_file[0].keys()])
			
			eye_blinks = [[((self.blinks_from_message_file[i]['start_timestamp']- start_time_scan)/1000) - nr_dummy_scans*tr, self.blinks_from_message_file[i]['duration']/1000,1] for i in range(len(self.blinks_from_message_file)) if (self.blinks_from_message_file[i]['start_timestamp']- start_time_scan) > (nr_dummy_scans*tr*1000)]
			
			
			saccades = [[((self.saccades_from_message_file[i]['start_timestamp']- start_time_scan)/1000) - nr_dummy_scans*tr, self.saccades_from_message_file[i]['duration']/1000,1] for i in range(len(self.saccades_from_message_file)) if np.all([(self.saccades_from_message_file[i]['start_timestamp']- start_time_scan) > (nr_dummy_scans*tr*1000), (self.saccades_from_message_file[i]['duration'] > saccade_duration_threshold)]) ]
			saccades_thresholded = [[((self.saccades_from_message_file[i]['start_timestamp']- start_time_scan)/1000) - nr_dummy_scans*tr, self.saccades_from_message_file[i]['duration']/1000,1] for i in range(len(self.saccades_from_message_file)) if np.all([(self.saccades_from_message_file[i]['start_timestamp']- start_time_scan) > (nr_dummy_scans*tr*1000), (self.saccades_from_message_file[i]['mystery_measure'] > mystery_threshold), (self.saccades_from_message_file[i]['duration'] > saccade_duration_threshold)]) ]
		
			np.savetxt(self.runFile(stage = 'processed/eye', run = r, extension = '.txt', postFix = ['eye_blinks']), np.array(eye_blinks), fmt = '%3.2f', delimiter = '\t')
			np.savetxt(self.runFile(stage = 'processed/eye', run = r, extension = '.txt', postFix = ['saccades']), np.array(saccades), fmt = '%3.2f', delimiter = '\t')
			np.savetxt(self.runFile(stage = 'processed/eye', run = r, extension = '.txt', postFix = ['saccades_thresholded']), np.array(saccades_thresholded), fmt = '%3.2f', delimiter = '\t')	
	

	###########################################################################################################################################
	######															Heart rate and blood pressure data									 ######
	######																												 				 ######		
	######																													 			 ######	
	###########################################################################################################################################	

	def physio(self):
		"""physio loops across runs to analyze their physio data"""
		for r in [self.runList[i] for i in self.conditionDict['WMM']]:
			pO = PhysioOperator(self.runFile(stage = 'processed/hr', run = r, extension = '.log' ))
			nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'] ))
			pO.preprocess_to_continuous_signals(TR = nii_file.rtime, nr_TRs = nii_file.timepoints)
	

	###########################################################################################################################################
	######															General linear model:												 ######
	######	Functions to run GLM, on single scans and across scans					 													 ######		
	######																													 			 ######	
	###########################################################################################################################################			

	def runAllGLMS(self):
		"""
		Take all transition events and use them as event regressors
		Run FSL on this
		"""
		for condition in ['WMM']:
			for run in self.conditionDict[condition]:
				
				# remove previous feat directories
				try:
					self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf', 'sgtf'], extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf', 'sgtf'], extension = '.feat'))
					os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf', 'sgtf'], extension = '.fsf'))
				except OSError:
					pass
				
				# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
				thisFeatFile = '/home/moorselaar/WMM_PRF/analysis/analysis.fsf'
				REDict = {
				#'---OUTPUT_DIR---':self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf']),
				'---NR_TRS---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf', 'sgtf'])).timepoints),
				'---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf', 'sgtf']), 
				'---CONFOUND_EV---':self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension='.par'), 
				# '---ANAT_FILE---':os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'bet', 'T1_bet' ), 
				'---STIM_FILE---':self.runFile(stage = 'processed/behavior', run = self.runList[run], postFix = ['stim_all'], extension='.txt'),
				'---RESPONSE_FILE---':self.runFile(stage = 'processed/behavior', run = self.runList[run], postFix = ['resp_all'], extension='.txt'),
				'---PPU_FILE---':self.runFile(stage = 'processed/hr', run = self.runList[run], postFix = ['ppu'], extension='.txt'),
				'---PPU_R_FILE---':self.runFile(stage = 'processed/hr', run = self.runList[run], postFix = ['ppu','raw'], extension='.txt'),
				'---RESP_FILE---':self.runFile(stage = 'processed/hr', run = self.runList[run], postFix = ['resp'], extension='.txt'),
				'---RESP_R_FILE---':self.runFile(stage = 'processed/hr', run = self.runList[run], postFix = ['resp','raw'], extension='.txt')
				}
				
				featFileName = self.runFile(stage = 'processed/mri', run = self.runList[run], extension = '.fsf')
				featOp = FEATOperator(inputObject = thisFeatFile)
				# no need to wait for execute because we're running the mappers after this sequence - need (more than) 8 processors for this, though.
				if self.runList[run] == [self.runList[i] for i in self.conditionDict['WMM']][-1]:
					featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
				else:
					featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
				self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
				# run feat
				featOp.execute()
	
	def setupRegistrationForFeat(self, wait_for_execute = True):
		"""apply the freesurfer/flirt registration for this session to a feat directory. This ensures that the feat results can be combined across runs and subjects without running flirt all the time."""
		for condition in ['WMM']:
			for run in self.conditionDict[condition]:
				
				feat_directory = self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf','sgtf'], extension='.feat')
				

				try:
					os.mkdir(os.path.join(feat_directory,'reg'))
				except OSError:
					pass
		
				if not os.path.isdir(self.stageFolder(stage = 'processed/mri/reg/feat/')):
					self.registerSession(prepare_register = True, bb = False, MNI = True)
		
				os.system('cp ' + self.stageFolder(stage = 'processed/mri/reg/feat/') + '* ' + os.path.join(feat_directory,'reg/') )
				if wait_for_execute:
					os.system('featregapply ' + feat_directory )
				else:
					os.system('featregapply ' + feat_directory + ' & ' )
	
	def gfeat_analysis(self, run_separate = True, run_combination = True):
		
		try:	# create folder
			os.mkdir(self.stageFolder('processed/mri/masks/stat/gfeat'))
			os.mkdir(self.stageFolder('processed/mri/masks/stat/gfeat/surf'))
		except OSError:
			pass
		for i in range(1,2): # all stats
			for stat in ['z','t','pe','cope']:
				
				afo = FlirtOperator( 	os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict['WMM'][0]]), 'combined/combined.gfeat', 'cope' + str(i) + '.feat', 'stats', stat + 'stat1.nii.gz'), 
										referenceFileName = self.runFile(stage = 'processed/mri/reg', base = 'forRegistration', postFix = [self.ID] )
										)
				# here I assume that the feat registration directory has been created. it's the files that have been used to create the gfeat, so we should be cool.
				afo.configureApply(		transformMatrixFileName = os.path.join(self.stageFolder('processed/mri/reg/feat/'), 'standard2example_func.mat'), 
										outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat/gfeat'), stat + str(i) + '_' + os.path.split(afo.inputFileName)[1]))
				afo.execute()
				# to surface
				stso = VolToSurfOperator(inputObject = afo.outputFileName)
				stso.configure(		frames = {'stat': 0} , 
									register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), 
									outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat/gfeat/surf'), os.path.split(afo.outputFileName)[1]))
				stso.execute()	

	# def gfeat_analysis(self, run_type='WMM', run_separate=True, run_combination=True):
		
	# 	# create folder
	# 	try:	
	# 		os.mkdir(self.stageFolder('processed/mri/masks/stat/gfeat'))
	# 		os.mkdir(self.stageFolder('processed/mri/masks/stat/gfeat/surf'))
	# 	except OSError:
	# 		pass
	# 	for i in range(1,2): # all stats
	# 		for stat in ['zstat1','tstat1','pe1','cope1']:
				
	# 			afo = FlirtOperator( 	os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), 'gfeat.gfeat', 'cope' + str(i) + '.feat', 'stats', stat + '.nii.gz'), 
	# 									referenceFileName = os.path.join(self.stageFolder(stage = 'processed/mri/WMM/3/'), self.dateCode + '_3_mcf.nii.gz')
	# 									)
	# 			# here I assume that the feat registration directory has been created. it's the files that have been used to create the gfeat, so we should be cool.
	# 			afo.configureApply(		transformMatrixFileName = os.path.join(self.stageFolder('processed/mri/reg/feat/'), 'standard2example_func.mat'), 
	# 									outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat/gfeat'), stat + str(i) + '_' + os.path.split(afo.inputFileName)[1]))
	# 			afo.execute()
	# 			# to surface
	# 			stso = VolToSurfOperator(inputObject = afo.outputFileName)
	# 			stso.configure(		frames = {'stat': 0} , 
	# 								register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), 
	# 								outputFileName = os.path.join(self.stageFolder('processed/mri/masks/stat/gfeat/surf'), os.path.split(afo.outputFileName)[1]))
	# 			stso.execute()



	###########################################################################################################################################
	######															Store and retrieve data from hdf5:									 ######
	######																			 													 ######		
	######																													 			 ######	
	###########################################################################################################################################	

	def remove_mask_stats(self):
		
		"If sum mask data == 0, delete this roi!"
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		for roi in anatRoiFileNames:
			file = NiftiImage(roi)
			if sum(file.data) == 0:
				print 'removed ' + roi
				os.system('rm ' + roi)
	
	def mask_stats_to_hdf(self, run_type='WMM', postFix=['mcf', 'sgtf'], sj_PRF_Session_folder = None):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results.
		- PER RUN: feat data, residuals, hpf_data, tf_data, and tf_psc_date.
		- COMBINED OVER RUNS: gfeat data, (polar and eccen data if polar_eccen==True)
			eccen data has to be in .../processed/mri/masks/eccen/eccen.nii.gz
			polar data has to be in .../processed/mri/masks/polar/polar.nii.gz
		"""
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
			
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if os.path.isfile(self.hdf5_filename):
			os.system('rm ' + self.hdf5_filename)
		self.logger.info('starting table file ' + self.hdf5_filename)
		h5file = openFile(self.hdf5_filename, mode = 'w', title = run_type + " file")
		# else:
		# 	self.logger.info('opening table file ' + self.hdf5_filename)
		# 	h5file = openFile(self.hdf5_filename, mode = "a", title = run_type + " file")
		
		######################################################################################################
		# ADD STATS PER RUN:
		for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
			"""loop over runs, and try to open a group for this run's data"""
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			
			try:
				thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data file ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
			"""
			Now, take different stat masks based on the run_type
			"""
			# this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf'], extension = '.feat')
			
			stat_files = {}
			for i in range(1,2):
				stat_files.update({
							'tstat' + str(i): os.path.join(this_feat, 'stats', 'tstat' + str(i) + '.nii.gz'),
							'zstat' + str(i): os.path.join(this_feat, 'stats', 'zstat' + str(i) + '.nii.gz'),
							'cope' + str(i): os.path.join(this_feat, 'stats', 'cope' + str(i) + '.nii.gz'),
							'pe' + str(i): os.path.join(this_feat, 'stats', 'pe' + str(i) + '.nii.gz'),
							})
			
			# general info we want in all hdf files
			stat_files.update({
								'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								'hpf_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								#'tf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf','psc','PRF']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'tf_psc_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'sgtf','psc','PRF']),
								})
			
			stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]

			for (roi, roi_name) in zip(rois, roinames):
				try:
					thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
					thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
				
				for (i, sf) in enumerate(stat_files.keys()):
					# loop over stat_files and rois
					# to mask the stat_files with the rois:
					try:
						imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
						these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
						h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
					except ZeroDivisionError:
						pass
		
		######################################################################################################
		# ADD COMBINED OVER RUNS STUFF
		
		#this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri/', extension = '_combined'))[1]
		
		#try:
		#	thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
		#	self.logger.info('data file ' + this_run_group_name + ' already in ' + self.hdf5_filename)
		#except NoSuchNodeError:
		#	# import actual data
		#	self.logger.info('Adding group ' + this_run_group_name + ' to this file')
		#	thisRunGroup = h5file.createGroup("/", this_run_group_name, ' imported from ' + self.runFile(stage = 'processed/mri/rivalry/combined/combined.gfeat', postFix = postFix))
		
		"""
		Now, take different stat masks based on the run_type
		"""
		#stat_files = {}
		#for i in range(1,28):
		#	stat_files.update({
		#				'tstat' + str(i): os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope' + str(i) + '_tstat1.nii.gz'),
		#				'zstat' + str(i): os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope' + str(i) + '_zstat1.nii.gz'),
		#				'cope' + str(i): os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope' + str(i) + '_cope1.nii.gz'),
		#				'pe' + str(i): os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/gfeat'), 'cope' + str(i) + '_pe1.nii.gz'),
		#				})
						
		#stat_files.update({
		#				'PRF_coef': os.path.join(self.stageFolder(stage = 'processed/mri/masks/PRF'), 'coefs_cortex_dilated_mask_mcf_sgtf_prZ_res_all_FUNC.nii.gz'),
		#				'PRF_corr': os.path.join(self.stageFolder(stage = 'processed/mri/masks/PRF'), 'corrs_cortex_dilated_mask_mcf_sgtf_prZ_res_all_FUNC.nii.gz'),
		#				'PRF_results': os.path.join(self.stageFolder(stage = 'processed/mri/masks/PRF'), 'results_cortex_dilated_mask_mcf_sgtf_prZ_res_all_FUNC.nii.gz'),
		#				})
		
		#stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
		
		#for (roi, roi_name) in zip(rois, roinames):
		#	try:
		#		thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
		#	except NoSuchNodeError:
				# import actual data
		#		self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
		#		thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
			
		#	for (i, sf) in enumerate(stat_files.keys()):
		#		# loop over stat_files and rois
		#		# to mask the stat_files with the rois:
		#		try:
		#			imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
		#			these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
		#			h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
		#		except ZeroDivisionError:
		#			pass	
		
		this_run_group_name = 'prf'
		try:
			thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			self.logger.info('data file already in ' + self.hdf5_filename)
		except NoSuchNodeError:
			# import actual data
			self.logger.info('Adding group ' + this_run_group_name + ' to this file')
			thisRunGroup = h5file.createGroup("/", this_run_group_name, '')
			
		stat_files = {}
		for c in ['all']:
			"""loop over runs, and try to open a group for this run's data"""
		
			"""
			Now, take different stat masks based on the run_type
			"""
			
			for res_type in ['results', 'coefs', 'corrs']:
				folder_name = os.path.join(sj_PRF_Session_folder, 'processed/mri/PRF/')
				filename = 'cortex_dilated_mask_mcf_sgtf_prZ_res_%s'%c
				stat_files.update({c+'_'+res_type: os.path.join(folder_name, res_type + '_' + filename + '.nii.gz')})
			
		
		stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
		
		for (roi, roi_name) in zip(rois, roinames):
			try:
				thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
				thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'ROI ' + roi_name +' imported' )
		
			for (i, sf) in enumerate(stat_files.keys()):
				# loop over stat_files and rois
				# to mask the stat_files with the rois:
				imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
				these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
				h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])

		h5file.close()
 		
	def roi_data_from_hdf(self, h5file, roi_wildcard, data_type, run = [], postFix = ['mcf','sgtf'],combined = False, prf = False):
		"""
		drags data from an already opened hdf file into a numpy array, concatenating the data_type data across voxels in the different rois that correspond to the roi_wildcard
		"""
		
		if combined == False:
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = postFix))[1]
		else:
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri/', extension = '_combined'))[1]

		if prf == True:
			this_run_group_name = 'prf'	
	
		try:
			thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			roi_names = []
			for roi_name in h5file.iterNodes(where = '/' + this_run_group_name, classname = 'Group'):
				if len(roi_name._v_name.split('.')) == 2:
					hemi, area = roi_name._v_name.split('.')
					if roi_wildcard == area:
						roi_names.append(roi_name._v_name)
				#if len(roi_name._v_name.split('.')) == 3:
				#	hemi, area, do_nothing = roi_name._v_name.split('.')
				#	if roi_wildcard == area:
				#		roi_names.append(roi_name._v_name)
			if len(roi_names) == 0:
				self.logger.info('No rois corresponding to ' + roi_wildcard + ' in group ' + this_run_group_name)
				return None
		except NoSuchNodeError:
			# import actual data
			self.logger.info('No group ' + this_run_group_name + ' in this file')
			return None
		
		all_roi_data = []
		for roi_name in roi_names:
			thisRoi = h5file.get_node(where = '/' + this_run_group_name, name = roi_name, classname='Group')
			all_roi_data.append( eval('thisRoi.' + data_type + '.read()') )
		all_roi_data_np = np.hstack(all_roi_data).T
		return all_roi_data_np


    ###########################################################################################################################################
	######															Deconvolution analysis:												 ######
	######					 													 														 ######		
	######																													 			 ######	
	###########################################################################################################################################	

	def deconvolve_roi(self, roi, threshold = 4.5, mask_type = 'center_Z', analysis_type = 'deconvolution', mask_direction = 'pos', signal_type = 'mean', runtype = 'WMM', sample_duration = 1):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		# check out the duration of these runs, assuming they're not all the same length.
		run_duration = []
		for r in [self.runList[i] for i in self.conditionDict[runtype]]:
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = r))
			tr, nr_trs = round(niiFile.rtime*1)/1000.0, niiFile.timepoints
			run_duration.append(tr * nr_trs)
		run_duration = np.r_[0,np.cumsum(np.array(run_duration))]

		conds = ['unique_patch_start','unique_patch_end','unique_center_start','unique_center_end']
		
		# cond_labels = ['attend_patch','attend_center']
		cond_labels = ['attend_patch_start','attend_patch_end','attend_center_start','attend_center_end']
		
		hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[runtype][0]]), runtype + '.hdf5')
		h5file = open_file(hdf5_filename, mode = 'r+', title = runtype + " file")
		
		event_data = []
		roi_data = []
		blink_events = []
		nuisance_data = []
		nr_runs = 0

		for r in [self.runList[i] for i in self.conditionDict[runtype]]:
			shell()
			roi_data.append(self.roi_data_from_hdf(h5file, roi[0],'tf_psc_data',r))
			
			this_blink_events = np.loadtxt(self.runFile(stage = 'processed/eye', run = r, extension = '.txt', postFix = ['eye_blinks']))
			this_blink_events += run_duration[nr_runs]
			blink_events.append(this_blink_events)

			nuisance_data.append(np.vstack([np.loadtxt(f).T for f in self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu']), self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['ppu_raw']), self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp']), self.runFile(stage = 'processed/hr', run = r, extension = '.txt', postFix = ['resp_raw']), self.runFile(stage = 'processed/mri', run = r, extension = '.par', postFix = ['mcf'])]).T)

			this_run_events = []
			for cond in conds:
				this_run_events.append(np.loadtxt(self.runFile(stage = 'processed/behavior', run = r, postFix = [cond], extension = '.txt'))[:,0])	# all unique trials are included (i.e. last trial is not removed)
			this_run_events = np.array(this_run_events) + run_duration[nr_runs]

			event_data.append(this_run_events)
			
			nr_runs += 1


		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T ) # mean intensity of every voxel is subtracted from every voxel value
		
		event_data_per_run = event_data
		roi_data_per_run = demeaned_roi_data
		
		roi_data = np.hstack(demeaned_roi_data)
		event_data = [np.concatenate([e[i] for e in event_data]) for i in range(len(event_data[0]))]

		# mapping data
		mapping_data = self.roi_data_from_hdf(h5file, roi[0],'zstat1',self.runList[self.conditionDict[runtype][0]])
		
		# thresholding of mapping data stat values
		if mask_direction == 'pos':
			mapping_mask = mapping_data[:,0] > threshold
		else:
			mapping_mask = mapping_data[:,0] < threshold
		
		#print roi_data.shape
		#print mapping_mask.sum()
		timeseries = eval('roi_data[mapping_mask,:].' + signal_type + '(axis = 0)')

		fig = pl.figure(figsize = (9, 5))
		s = fig.add_subplot(211)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		time_signals = []
		if analysis_type == 'deconvolution':
			interval = [0.0,13.5]
			# nuisance version?
			
			nuisance_design = Design(timeseries.shape[0], tr) 
			nuisance_design.configure(np.array([list(np.vstack(blink_events))]))

			nuisance_design_matrix = nuisance_design.designMatrix
			nuisance_design_matrix = np.vstack((nuisance_design_matrix, np.vstack(nuisance_data).T)).T
			nuisance_design_matrix = np.repeat(nuisance_design_matrix, sample_duration, axis = 0)

			deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr, deconvolutionInterval = interval[1], run = False)
			#deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr, deconvolutionInterval = interval[1], run = True)
			deco.runWithConvolvedNuisanceVectors(nuisance_design.designMatrix)

			for i in range(0, deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[0]):
				time_signals.append(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze())
				# shell()
				pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventTypeNuisance.shape[1]), np.array(deco.deconvolvedTimeCoursesPerEventTypeNuisance[i].squeeze()), ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			
			# the following commented code doesn't factor in blinks as nuisances
			# deco = DeconvolutionOperator(inputObject = timeseries, eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
			# for i in range(0, deco.deconvolvedTimeCoursesPerEventType.shape[0]):
			# 	pl.plot(np.linspace(interval[0],interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), deco.deconvolvedTimeCoursesPerEventType[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
			# 	time_signals.append(deco.deconvolvedTimeCoursesPerEventType[i])
			
			s.set_title('deconvolution' + roi[0] + ' ' + mask_type)
		
		else:
			interval = [-3.0,19.5]
			# zero_timesignals = eraO = EventRelatedAverageOperator(inputObject = np.array([timeseries]), eventObject = event_data[0], interval = interval)
			# zero_time_signal = eraO.run(binWidth = 3.0, stepSize = 1.5)
			for i in range(event_data.shape[0]):
				eraO = EventRelatedAverageOperator(inputObject = np.array([timeseries]), eventObject = event_data[i], TR = tr, interval = interval)
				time_signal = eraO.run(binWidth = 3.0, stepSize = 1.5)
				zero_zero_means = time_signal[:,1] - time_signal[time_signal[:,0] == 0,1]
				s.fill_between(time_signal[:,0], zero_zero_means + time_signal[:,2]/np.sqrt(time_signal[:,3]), zero_zero_means - time_signal[:,2]/np.sqrt(time_signal[:,3]), color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])
				pl.plot(time_signal[:,0], zero_zero_means, ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
				time_signals.append(zero_zero_means)
			s.set_title('event-related average ' + roi + ' ' + mask_type)
		
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]+1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		
		s = fig.add_subplot(212)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		if analysis_type == 'deconvolution':
			for i in range(0, len(event_data)/2):
				
				ts_diff = -(time_signals[i] - time_signals[i+2])
				pl.plot(np.linspace(0,interval[1],deco.deconvolvedTimeCoursesPerEventType.shape[1]), np.array(ts_diff), ['k','r'][i], label = ['start','end'][i]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
				s.set_title('attend_signal' + roi[0] + ' ' + mask_type + ' ' + analysis_type)
		
		else:
			time_signals = np.array(time_signals)
			for i in range(0, event_data.shape[0], 2):
				ts_diff = -(time_signals[i] - time_signals[i+1])
				pl.plot(time_signal[:,0], ts_diff, ['b','b','g','g'][i], alpha = [1.0, 0.5, 1.0, 0.5][i], label = ['fixation','visual stimulus'][i/2]) #  - time_signal[time_signal[:,0] == 0,1] ##  - zero_time_signal[:,1]
			s.set_title('reward signal ' + roi + ' ' + mask_type + ' ' + analysis_type)
		
		s.set_xlabel('time [s]')
		s.set_ylabel('$\Delta$ % signal change')
		s.set_xlim([interval[0]-1.5, interval[1] + 1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
			
		h5file.close()
		# mapper_h5file.close()
		
		pl.draw()
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi[0] + '_' + mask_type + '_' + mask_direction + '_' + analysis_type + '.pdf'))
		
		return [roi[0] + '_' + mask_type + '_' + mask_direction + '_' + analysis_type, event_data, timeseries, np.array(time_signals)]
	

	###########################################################################################################################################
	######															Raw input analysis:													 ######
	######	Simple correlation between stimulus input and brain data as a first step to check whether decoding based on a feedforward	 ######	
	######	model is possible.		 																									 ######		
	######																													 			 ######	
	###########################################################################################################################################

	def rescale_images(self, patches = [], n_pixel_elements = 42, flip = True, save = False):
		"""
		function to adjust nr of stimulus elements to match prf data
		"""			
		
		rescaled_patches = []
		for i in patches:
			patch = plt.imread(os.path.join(self.project.base_dir,'patches','image'+str(i)+'.png'))
			if flip == True:
				patch = np.flipud(patch)
			patch = patch[:,patch.shape[1]/2 - patch.shape[0]/2:patch.shape[1]/2 + patch.shape[0]/2,0] # visual field 1080 by 1080
			
			scaled_patch = []
			scale = patch.shape[0]/n_pixel_elements
			for x in range(n_pixel_elements):
				for y in range (n_pixel_elements):
					# michelson_contrast
					scaled_patch.append(np.max(patch[scale*x:scale*x + scale,scale*y:scale*y + scale]) - np.min(patch[scale*x:scale*x + scale,scale*y:scale*y + scale]))
			scaled_patch = np.asarray(scaled_patch).reshape([n_pixel_elements,n_pixel_elements],order = 'C')
			if save == True:
				imshow(scaled_patch)
				plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'heatmap_patch' + str(i)))	
			rescaled_patches.append(scaled_patch)

		return rescaled_patches

	def create_stim_mask(self, stim_shape = [], inner = True, radius_inner = 1, outer = True, radius_outer = 0):
		"""
		Stim mask returns boolean array that can be used to mask input stimuli.
		Radius_inner and radius_outer range from 0-1, where 1 includes the whole image.
		"""	

		#create stim mask (based on eccentricity mask and atend centre mask)
		centre_x, centre_y = stim_shape/2,stim_shape/2

		if outer:
			y,x = np.ogrid[-centre_x:stim_shape-centre_x, -centre_y:stim_shape-centre_y]
			mask_ecc = x*x + y*y <= (radius_outer*stim_shape/2)**2

		if inner:	
			y,x = np.ogrid[-centre_x:stim_shape-centre_x, -centre_y:stim_shape-centre_y]
			mask_centre = x*x + y*y >= (radius_inner*stim_shape/2)**2

		if inner and outer:
			mask_stim = (mask_ecc.ravel()*mask_centre.ravel())
		elif inner and not outer:
			mask_stim = mask_centre.ravel()
		elif outer and not inner:
			mask_stim = mask_ecc.ravel()

		return mask_stim



	###########################################################################################################################################
	######															Extra scripts:													     ######
	######		 																									 					 ######		
	######																													 			 ######	
	###########################################################################################################################################

	def setup_all_data_for_predict(self, roi, contrast, run_type='WMM', postFix=['mcf','sgtf']):
		"""
		function that reads out and returns all relevant HDF5 data that is necessary for decoding patches
		"""
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		h5file = openFile(self.hdf5_filename, mode = 'r+', title = run_type + " file")	
		
		# Load all functional data (per run) and load all combined data:
		roi_dict = {}
		roi_data_per_roi = []
		mask_data_per_roi = []

		for j in range(len(roi)):
			patch = 1
			mask_data = []
			contrast_dict = {}
			combined_dict = {}
			roi_data_PFR = self.roi_data_from_hdf(h5file, roi_wildcard = roi[j], data_type = 'PRF_coef', combined = True).squeeze()
			for contr_nr in range(1,25): # 24 relevant contrasts; 12 patches vs baseline \ 12 patches vs stimulation
				roi_data_runs = []
				roi_data_comb = []
				for r in [self.runList[i] for i in self.conditionDict[run_type]]:
					if roi_data_runs == []:
						roi_data_comb.append(self.roi_data_from_hdf(h5file, roi_wildcard = roi[j], data_type = contrast + str(contr_nr), combined = True).squeeze())
					roi_data_runs.append(self.roi_data_from_hdf(h5file, roi[j], contrast + str(contr_nr), run = r).squeeze()) # squeeze()
					
				
				if contr_nr % 2 == 1:
					contrast_dict.update({'base_con' + str(patch) : np.hstack(roi_data_runs)})
					contrast_dict.update({'base_con_comb' + str(patch) : np.hstack(roi_data_comb)})
				elif contr_nr % 2 == 0:
					contrast_dict.update({'stim_con' + str(patch) : np.hstack(roi_data_runs)})
					contrast_dict.update({'stim_con_comb' + str(patch) : np.hstack(roi_data_comb)})
					patch += 1
				if contr_nr == 24:
					contrast_dict.update({'PRF_coef': roi_data_PFR})
					

			roi_dict.update({roi[j]:contrast_dict})	
		
		return roi_dict

	def predict_patches(self,roi, contrast, nr_of_elements = 60, masked = True):
		"""
		function that returns a dictionary with scalar arrays for each individual patch
		"""
		
		ROI_data_all = self.setup_all_data_for_predict(roi, contrast)
		
		roi_dict = {}

		# boolean (nr_of_elements,nr_of_elements) mask to remove information that is outside the patch
		if masked == True:
			y, x = np.ogrid[(nr_of_elements/2)*-1: (nr_of_elements/2),(nr_of_elements/2)*-1: (nr_of_elements/2)]
			mask = x**2 + y**2 <= (nr_of_elements/2)**2
			mask = mask.reshape(nr_of_elements*nr_of_elements)
			index_to_remove = np.where(mask == False)
		else:
			index_to_remove = []
		
		stimulus_data = self.rescale_images()
		for j in range(len(roi)):
			PRF_data_roi =  ROI_data_all[roi[j]]['PRF_coef']
			patch_dict = {}
			
			for patch in range(12):
				factor = []
				stimulus = stimulus_data[patch].reshape(3600,)
				stimulus = np.delete(stimulus, index_to_remove)
				for voxel in range(PRF_data_roi.shape[0]):
					PRF = PRF_data_roi[voxel]
					PRF = np.delete(PRF, index_to_remove)
					factor.append(np.array([ np.dot(PRF, stimulus)/np.dot(PRF,PRF)]))
				
				patch_dict.update({'patch_' + str(patch): np.hstack(factor)})

			roi_dict.update({roi[j]: patch_dict})
				
		self.ROI_data_all = ROI_data_all
		self.ROI_PRF_norm = roi_dict
		self.contrast_type = contrast




			


	def check_contrast_values(self, contrast = 'stim', normalized = False, save = False):
		"""
		function that returns scatter plots of beta values across runs. Check whether data is correlated for each patch across runs and uncorrelated for all patches across runs
		"""
	
		# check each patch separately across runs 
		comparisons = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
		for i in range(12):
			if contrast == 'stim':
				dd = self.ROI_data_all['V1']['stim_con%i'%(i+1)].reshape((4, self.ROI_data_all['V1']['stim_con%i'%(i+1)].shape[0]/4)) 	
			elif contrast == 'base':
				dd = self.ROI_data_all['V1']['base_con%i'%(i+1)].reshape((4, self.ROI_data_all['V1']['base_con%i'%(i+1)].shape[0]/4))
			counter = 0	
			for c in comparisons:	
				ax = plt.subplot(12,6,((i*6) + counter))
				ax.scatter(dd[c[0]],dd[c[1]],label = str(round(pearsonr(dd[c[0]],dd[c[1]])[0],2)))
				ax.legend(loc = 0)
				ax.set_title(str(c[0]) + '-' + str(c[1]))
				counter += 1				
		if save == True:
			plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), self.contrast_type))
		

		# check correlations across patches 
		coef_matrix = []
		fig, axes = plt.subplots(2,2)
		for run in range(4):	
	
			cor_matrix = []
			ax = axes.flat[run]

			for patch in range(12):
				if contrast == 'stim':
					cor_matrix.append(self.ROI_data_all['V1']['stim_con%i'%(patch+1)].reshape((4, self.ROI_data_all['V1']['stim_con%i'%(patch+1)].shape[0]/4))[run])
				elif contrast == 'base':
					cor_matrix.append(self.ROI_data_all['V1']['base_con%i'%(patch+1)].reshape((4, self.ROI_data_all['V1']['base_con%i'%(patch+1)].shape[0]/4))[run])	
			
			if normalized == False:			
				im = ax.imshow(np.corrcoef(cor_matrix),  interpolation = 'nearest', vmin = -1, vmax = 1)
				ax.set_title("run " + str(run + 1))

			else:
				# zscore per voxel across patches
				cor_matrix = np.array(cor_matrix)
				voxel_to_remove = [i for i in range(cor_matrix.shape[1]) if np.sum(np.where(cor_matrix[:,i] == 0)) > 0] # voxels that contain a value of 0 for one of the patches
				cor_matrix = np.delete(cor_matrix,voxel_to_remove,axis = 1)
				cor_matrix = (cor_matrix- cor_matrix.mean(axis = 0))/cor_matrix.std(axis=0) 
				im = ax.imshow(np.corrcoef(cor_matrix),  interpolation = 'nearest', vmin = -1, vmax = 1)
				ax.set_title("run " + str(run + 1))

			# store output of all runs
			coef_matrix.append(np.corrcoef(cor_matrix))

		# finish figure with color_bar
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.87])
		fig.colorbar(im, cax=cbar_ax)
		
		# save figures (4 subbplots) 
		if normalized == False:
			if save == True:
				plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_' + contrast + "_" + self.contrast_type))
		else:
			if save == True:
				plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_nor_' + contrast + "_" + self.contrast_type))

		
		# check correlations across patches for all runs combined (gfeat)
		cor_matrix_comb = []
		plt.figure()
		for patch in range(12):
			if contrast == 'stim':
				cor_matrix_comb.append(self.ROI_data_all['V1']['stim_con_comb%i'%(patch+1)])
			elif contrast == 'base':
				cor_matrix_comb.append(self.ROI_data_all['V1']['base_con_comb%i'%(patch+1)])

		if normalized == False:	
			plt.imshow(np.corrcoef(cor_matrix_comb),  interpolation = 'nearest', vmin = -1, vmax = 1)
			plt.title("combined")
			plt.colorbar(ticks = [-1,0,1])
		else:	
			cor_matrix_comb = np.array(cor_matrix_comb)
			voxel_to_remove = [i for i in range(cor_matrix_comb.shape[1]) if np.sum(np.where(cor_matrix_comb[:,i] == 0)) > 0]
			cor_matrix_comb = np.delete(cor_matrix_comb,voxel_to_remove,axis = 1)
			cor_matrix_comb = (cor_matrix_comb- cor_matrix_comb.mean(axis = 0))/cor_matrix_comb.std(axis=0)
			plt.imshow(np.corrcoef(cor_matrix_comb),  interpolation = 'nearest', vmin = -1, vmax = 1)
			plt.title('combined') 
			plt.colorbar(ticks = [-1,0,1])

		if save == True:
			if normalized == False:
				plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_comb_' + contrast + "_" + self.contrast_type))
			else:
				plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_comb_nor_' + contrast + "_" + self.contrast_type))
		else:
			return coef_matrix, np.corrcoef(cor_matrix_comb)
	
	def compare_stim_contrast(self, contrast = 'stim', normalized = False):
		"""
		function that compares stimuli heatmap to FSL_contrast estimates by subtracting both correlation matrices
		"""
		
		stimuli = self.compare_stimuli()
		output_runs, output_comb = self.check_contrast_values(contrast = contrast, normalized = normalized)

		fig, axes = plt.subplots(2,2)
		for run in range(len(output_runs)):
			ax = axes.flat[run]
			im = ax.imshow(((1 -abs(stimuli - output_runs[run]))), interpolation = 'nearest', vmin = -1, vmax = 1)
			ax.set_title("run " + str(run + 1) + ": " + str(round((1 - abs(stimuli - output_runs[run])).mean(),3)))
		
		# finish figure with color_bar
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.05, 0.05, 0.87])
		fig.colorbar(im, cax=cbar_ax)
		

		if normalized == False:
			plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'comp_stim_' + contrast + "_" + self.contrast_type))
		else:
			plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'comp_stim_' + contrast + '_nor' + "_" + self.contrast_type))	

		plt.figure()
		plt.imshow(((1 -abs(stimuli - output_comb))), interpolation = 'nearest', vmin = -1, vmax = 1)
		plt.title('combined: ' + str(round((1 - abs(stimuli - output_comb)).mean(),3)))
		plt.colorbar(ticks = [-1,0,1])

		if normalized == False:
			plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'comp_stim_comb_' + contrast + "_" + self.contrast_type))
		else:
			plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'comp_stim_comb' + contrast + '_nor' + "_" + self.contrast_type))	

	
	def PRF_animation(self, contrast, cut_off = 1.5):
		
		# contrast values for all patches
		patch_contrasts = []
		for patch in range(12):
			if contrast == 'stim':
				patch_contrasts.append(self.ROI_data_all['V1']['stim_con%i'%(patch+1)].reshape((4, self.ROI_data_all['V1']['stim_con%i'%(patch+1)].shape[0]/4)))
			elif contrast == 'base':
				patch_contrasts.append(self.ROI_data_all['V1']['base_con%i'%(patch+1)].reshape((4, self.ROI_data_all['V1']['stim_con%i'%(patch+1)].shape[0]/4)))
		

		# PRF values for all voxels
		PRF=self.ROI_data_all['V1']['PRF_coef']

		# normalize PRF
		PRF = (PRF- PRF.mean(axis = 0))/PRF.std(axis=0)


		# for each run, separately predict how brain sees stimuli
		for run in range(4):
			fig, axes = plt.subplots(4,3)
			for patch in range(12):
				ax = axes.flat[patch]
				ax.imshow(np.mean([PRF[voxel]*patch_contrasts[patch][run][voxel] for voxel in np.where(abs(patch_contrasts[patch][run])>abs(patch_contrasts[patch][run]).max()-cut_off)[0].tolist()],axis = 0).reshape(60,60))
				ax.set_title("patch_" + str(patch))

			plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'PRF_fit_' + str(run) + '_' + contrast + "_" + self.contrast_type))
	



	
	# Step 1: identify patches that are most similar and identify patches that are most dissimilar
	def compare_stim_input(self, rescale_PRF = False, save = False):
		"""
		function that returns correlation matrix of rescaled patches
		"""	

		# first check which (unique) patches are presented
		stim_info = np.loadtxt(self.runFile(stage = 'processed/behavior', extension = '.txt', postFix = ['stim_info_all']))
		unique_stim = np.unique(stim_info[:,2])
		stimuli = np.array(self.rescale_images(patches = list(np.array(unique_stim, dtype = int))))

		if rescale_PRF:
			stim = self.rescale_images()
		else:
			shell()
			stim	


		stim_array = []
		for i in range(len(stimuli)):
			stim_array.append(stimuli[i].reshape(3600))
		plt.imshow(np.corrcoef(stim_array),  interpolation = 'nearest', vmin = -1, vmax = 1)
		plt.colorbar(ticks = [-1,0,1])

		if save == True: 
			plt.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_matrix_heatmap'))
		else:
			return np.corrcoef(stim_array) 


	def compare_brain_and_stimulus_input_data(self):
		"""
		First step in the PRF analysis. Check whether stimuli that have different features (different spatial patterns) values also have different 
		activation patterns in early visual areas.
		This is a criucial condition for the PRF analysis to be meaningfull
		"""	

		# first check which (unique) patches are presented
		stim_info = np.loadtxt(self.runFile(stage = 'processed/behavior', extension = '.txt', postFix = ['stim_info_all']))
		unique_stim = np.unique(stim_info[:,2])
		stimuli = np.array(self.rescale_images(patches = list(np.array(unique_stim, dtype = int))))

		# select patches that are similar and patches that are dissimilar
		mask = self.create_stim_mask(stim_shape = stimuli.shape[1], radius_inner = 0.04, outer = False)
		
		# create correlation matrix for all presented stimuli
		stim_correlations = np.zeros((stimuli.shape[0], stimuli.shape[0]))
			
		for i, stim in enumerate(stimuli): # all stimuli
			stim_correlations[i, :] =  np.array([pearsonr(stim.ravel()[mask], stim_2.ravel()[mask])[0] for stim_2 in stimuli])

		# store similarity info per individual stimuli [stimuli, less similar, most, similar]		
		stim_similarity = np.array([[unique_stim[i], unique_stim[np.argsort(stim_correlations[i])[0]],  unique_stim[np.argsort(stim_correlations[i])[-2]]] for i in range(stim_correlations.shape[0])])		


		stim_correlations[i][np.argsort(stim_correlations[i])[0]], stim_correlations[i][np.argsort(stim_correlations[i])[-2]]
	





	def pca_stim_input (self, unique_stim = 100, nr_components = 10):
		"""
		function that applies PCA on stimulus data to condense stimulus information into a smaller set of new composite dimensions
		"""		

		# create N X M matrix (N is number of stimuli, M is number of pixels), only include informative pixels (pixels inside patch)
		stim_matrix = []
		for i in range(unique_stim):
			if i == 0:
				shape = plt.imread(os.path.join(self.project.base_dir,'patches','image'+str(i)+'.png')).shape
			stim = plt.imread(os.path.join(self.project.base_dir,'patches','image'+str(i)+'.png'))[:,:,0].ravel()
			stim_matrix.append(stim)
		stim_matrix = np.vstack(stim_matrix)	
		

		# normalize the data
		stim_matrix = sklearn.preprocessing.normalize(stim_matrix,axis = 0)
		#stim_matrix = (stim_matrix- stim_matrix.mean(axis = 0))/stim_matrix.std(axis=0)

		# PCA
		pca = PCA(n_components = nr_components)
		pca.fit(stim_matrix)

		# plot components
		for i in range(nr_components):
			ax = plt.subplot(5,2,i)
			ax.set_title('Component_' + str(i) + ': '+ str(pca.explained_variance_ratio_[i]))
			ax.imshow(pca.components_[i].reshape(shape[0],shape[1]))

		plt.show()	

	def pcaPlot (self, contrast_type = 'stim'):

		# patch order
		order = np.argsort([int(p.split('_')[-1]) for p in self.ROI_PRF_norm['V1'].keys()])
		stim_array = np.array([self.ROI_PRF_norm['V1']['patch_%i'%d] for d in order])

		# normalize data
		#stim_array = sklearn.preprocessing.normalize(stim_array,axis = 0)
		stim_array = (stim_array- stim_array.mean(axis = 0))/stim_array.std(axis=0)


		# principal component analysis with 5 components on stim_array (nr_of patches by number of voxels)
		pc = PCA(n_components = 5)
		pc.fit(stim_array)

		#transform data
		for i in range(12):
			if contrast_type == 'stim':
				dd = self.ROI_data_all['V1']['stim_con%i'%(i+1)].reshape((4, self.ROI_data_all['V1']['base_con%i'%(i+1)].shape[0]/4))
			elif contrast_type == 'base':
				dd = self.ROI_data_all['V1']['base_con%i'%(i+1)].reshape((4, self.ROI_data_all['V1']['stim_con%i'%(i+1)].shape[0]/4))
			pc.fit(dd)

			ax = plt.subplot(4,3,i)
			ax.set_title(str(pc.explained_variance_ratio_[0]))
			ax.plot(pc.transform(dd))

		plt.show()


	def decode_patches_from_roi(self, nr_patches = 100, roi = 'V1', stim_stat_threshold = 2.0, prf_stat_threshold = 4.0, prf_ecc_threshold = 0.6, run_type = 'WMM', smoothing_for_PRF = 5, scaling = True, sum_TRs = True,summed_TRs = 3):
		"""decode_patches_from_roi takes the name of ROI, takes the most active voxels' PRFs from this ROI 
		and uses this PRF to predict the stimulus shown on a given trial. This is then assessed by a simple correlation for now.
		"""
		
		# first determine individual run duration (to make sure that stimulus timings of all runs are correct)
		run_duration = []
		for r in [self.runList[i] for i in self.conditionDict['WMM']]:
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = r))
			tr, nr_trs = round(niiFile.rtime*1)/1000.0, niiFile.timepoints
			run_duration.append(tr * nr_trs)
		run_duration = np.r_[0,np.cumsum(np.array(run_duration))]

		# timing information stimuli
		patches_all_trials = []
		run = 0
		for r in [self.runList[i] for i in self.conditionDict['WMM']]:
			stim_events = np.loadtxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = ['stim' ,'all','task']))
			patches_run= np.array([[stim_events[i][2],np.ceil((stim_events[i][0]+ run_duration[run])/tr),np.ceil((stim_events[i][1] +run_duration[run])/tr),stim_events[i][3]] for i in range(stim_events.shape[0])])
			patches_all_trials.append(patches_run)
			run += 1

		patches_all_trials = np.vstack(patches_all_trials) # array (patches, timing_memory, timing test)
		unique_patches = np.unique(patches_all_trials[:,0]) # array (unique_patches)

		# numpy array of all patches from stimulus pool (also rotated by 90 degrees to check orientation space of PRF)
		patches = np.array(self.rescale_images())
		patches_rotated = [patches]
		for i in range(3):
			patches_rotated.append(np.array([np.rot90(p) for p in patches_rotated[-1]]))
		patches_rotated = np.array(patches_rotated) 
		
		# brain data
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		self.logger.info('data from table file ' + self.hdf5_filename)
		h5file = open_file(self.hdf5_filename, mode = 'r')		

		stim_stats = self.roi_data_from_hdf(h5file, roi, data_type = 'zstat1', run = self.runList[self.conditionDict[run_type][0]], postFix = ['mcf','sgtf'], combined = False, prf = False)
		prfs = self.roi_data_from_hdf(h5file, roi, data_type = 'all_coefs', run = self.runList[self.conditionDict[run_type][0]], postFix = ['mcf','sgtf'], combined = False, prf = True)
		prf_stats = self.roi_data_from_hdf(h5file, roi, data_type = 'all_corrs', run = self.runList[self.conditionDict[run_type][0]], postFix = ['mcf','sgtf'], combined = False, prf = True)
		
		prf_ecc = self.roi_data_from_hdf(h5file, roi, data_type = 'all_results', run = self.runList[self.conditionDict[run_type][0]], postFix = ['mcf','sgtf'], combined = False, prf = True)[:,1]

		timeseries = []
		for i in range(len(self.conditionDict[run_type])):
			timeseries.append(self.roi_data_from_hdf(h5file, roi, data_type = 'tf_psc_data', run = self.runList[self.conditionDict[run_type][i]], postFix = ['mcf','sgtf'], combined = False, prf = False))
		timeseries = np.concatenate(timeseries, axis = 1) 

		# select voxels based on prf and stimulus contrasts
		prfs = prfs[((stim_stats.squeeze() > stim_stat_threshold) * (prf_stats[:,1] > prf_stat_threshold)*(prf_ecc < prf_ecc_threshold)).squeeze(),:] 
		timeseries = timeseries[((stim_stats.squeeze() > stim_stat_threshold) * (prf_stats[:,1] > prf_stat_threshold)*(prf_ecc < prf_ecc_threshold)).squeeze(),:] 
		prf_stats = prf_stats[((stim_stats.squeeze() > stim_stat_threshold) * (prf_stats[:,1] > prf_stat_threshold)*(prf_ecc < prf_ecc_threshold)).squeeze(),1]

		#create stim mask (based on eccentricity mask and atend centre mask)
		centre_x, centre_y = patches.shape[1]/2,patches.shape[1]/2

		y,x = np.ogrid[-centre_x:patches.shape[1]-centre_x, -centre_y:patches.shape[1]-centre_y]
		mask_ecc = x*x + y*y <= (prf_ecc_threshold*patches.shape[1]/2)**2

		y,x = np.ogrid[-centre_x:patches.shape[1]-centre_x, -centre_y:patches.shape[1]-centre_y]
		mask_centre = x*x + y*y >= 4**2

		mask_stim = (mask_ecc.ravel()*mask_centre.ravel())	

		# create scaling matrix (identity matrix) of nr_voxels by nr_voxels
		if scaling:
			scaling_matrix = np.eye(prfs.shape[0])
			for i in range(len(prf_stats)):
				scaling_matrix[i,i] = prf_stats[i]


		# based on prfs and scaling factor create brain patches (either for single TRs or for summed TRS)		
		brain_patches = np.mat(timeseries.T) * np.mat(scaling_matrix) * np.mat(prfs) 
		if sum_TRs == True:
			timeseries_summed = np.array([np.sum(timeseries.T[i:i+summed_TRs,:],axis = 0) for i in range(timeseries.shape[1]) if i < timeseries.shape[1] - summed_TRs])
			brain_patches= np.mat(timeseries_summed) * np.mat(scaling_matrix) * np.mat(prfs)
       
		# per TR (either summed or individual) calculate correlations between brain patch and stimulus input
		patch_correlations = np.zeros((4, brain_patches.shape[0], patches.shape[0]))

		for k in range(4): # 4 orientations
			for i, t in enumerate(np.array(brain_patches)): # all brain_patches
				patch_correlations[k, i, :] =  np.array([pearsonr(t[mask_stim], patch.ravel()[mask_stim])[0] for patch in patches_rotated[k]])

		# for analysis only look at correlations for all patches that were presented in this subjects session
		patch_index_unique = np.array([j for i in patches_all_trials[:,0] for j in range(len( np.unique(patches_all_trials[:,0]))) if i == np.unique(patches_all_trials[:,0])[j]])	
		patch_correlations_unique = patch_correlations[:,:,np.array(unique_patches, dtype = int)]

		# show correlation matrix of patches for all rotations (dots represent stimulus that was present)
		fig, axes = plt.subplots(2,2)
		for i in range(patches_rotated.shape[0]):
			ax = axes.flat[i]
			ax.imshow(patch_correlations_unique[i,:,:].T,vmin = np.min(patch_correlations_unique), vmax = np.max(patch_correlations_unique))
			ax.scatter(patches_all_trials[:,1],patch_index_unique)
			ax.set_title("rotation_" + str(i*90))

		# show decoding evidence across runs for all unique patches	
		fig, axes = plt.subplots(2,2)	
		for i in range(patches_rotated.shape[0]):
			ax = axes.flat[i]
			ax.hist(np.argmax(patch_correlations_unique[i,:,:], axis = 1), bins = patch_correlations_unique.shape[2] + 1)
			ax.set_ylim([0,600])
			ax.set_title("rotation_" + str(i*90))
		
		# show brain_patches and stimulus input		
		mask_stim = np.array([not i for i in mask_stim]).reshape(patches.shape[1],patches.shape[1]) # flip stim mask for graphing purposes

		index_att_patch = patches_all_trials[:,3] == 0
		index_att_center = patches_all_trials[:,3] == 1 


		# calculate decoding performance
		correlations_unique_TR_patch = np.array([patch_correlations[:,np.array(patches_all_trials[:,1],dtype = int)[index_att_patch]+ TR][:,:,np.array(unique_patches, dtype = int)] for TR in range(2,5)])
		correlations_unique_TR_center = np.array([patch_correlations[:,np.array(patches_all_trials[:,1],dtype = int)[index_att_center]+ TR][:,:,np.array(unique_patches, dtype = int)] for TR in range(2,5)])

		# check relative position of presented patch compared to other unique patches
		patch_dict = {}
		center_dict = {}
		for condition in ['patch' ,'center']:
			for TR in range(correlations_unique_TR_patch.shape[0]):
				for rotation in range(correlations_unique_TR_patch.shape[1]):
					if condition == 'patch':
						print 'update'
						decode = np.array([int(np.where(np.argsort(correlations_unique_TR_patch[TR,rotation,i])==patch_index_unique[index_att_patch][i])[0]) for i in range(correlations_unique_TR_patch.shape[2])])
						patch_dict.update({'RT_' + str(TR) + '_Rotate_' + str(rotation):[decode, np.mean(decode),np.where(decode == np.max(decode))]})
					elif condition == 'center':	
						decode = np.array([int(np.where(np.argsort(correlations_unique_TR_center[TR,rotation,i])==patch_index_unique[index_att_patch][i])[0]) for i in range(correlations_unique_TR_patch.shape[2])])
						center_dict.update({'RT_' + str(TR) + '_Rotate_' + str(rotation):[decode, np.mean(decode),np.where(decode == np.max(decode))]})

		shell()
		# show decoding performance
		for TR in [2]:
			correlations_unique_TR_patch = patch_correlations[:,np.array(patches_all_trials[:,1],dtype = int)[index_att_patch]+ TR][:,:,np.array(unique_patches, dtype = int)]
			correlations_unique_TR_cent = patch_correlations[:,np.array(patches_all_trials[:,1],dtype = int)[index_att_cent]+ TR][:,:,np.array(unique_patches, dtype = int)]

			for cond in ['patch','center']:
				fig, axes = plt.subplots(2,2)
				for k in range(4): # 4 rotations
					ax = axes.flat[k]
					ax.set_title("rotation_" + str(k*90))
					if cond == 'patch':
						ax.hist([np.argsort(correlations_unique_TR_patch[k][i])[patch_index_unique[i]] for i in range(len(patch_index_unique)/2)], alpha = 0.3, color = ['r','g','b','k'][k], bins = 44, cumulative = True, histtype = 'step')
					elif cond == 'center':
						ax.hist([np.argsort(correlations_unique_TR_cent[k][i])[patch_index_unique[i]] for i in range(len(patch_index_unique)/2)], alpha = 0.3, color = ['r','g','b','k'][k], bins = 44, cumulative = True, histtype = 'step')

		if sum_TRs == True:
			bpr = np.array(brain_patches).reshape((timeseries.shape[1] - summed_TRs,patches.shape[1],patches.shape[1])) # brain patch reshaped to PRF space, length is nr of TRS
		else:
			bpr = np.array(brain_patches).reshape((timeseries.shape[1],patches.shape[1],patches.shape[1]))
			
		
		tr_bpr_patch = np.array([bpr[np.array(patches_all_trials[:,1], dtype = int)[index_att_patch]+i] for i in range(2,5)]) #brain patches for different timepoints 
		tr_bpr_cent = np.array([bpr[np.array(patches_all_trials[:,1], dtype = int)[index_att_cent]+i] for i in range(2,5)])

		
		# show decoding attend patches condition
	


		patch = 0
		test = [i for i in range(index_att_patch.shape[0]) if index_att_patch[i]]
		for i in test:
			print i, patch
			f = pl.figure()
			patch += 1
			#f = pl.figure()
			#s = f.add_subplot(241, aspect = 'equal')
			
			#tr_bpr_patch[0,patch][mask_stim] = 0
			#pl.imshow(tr_bpr_patch[0,patch],vmin = np.min(tr_bpr), vmax = np.max(tr_bpr))
			#s = f.add_subplot(242, aspect = 'equal')
			#pl.imshow(tr_bpr_patch[0,patch],vmin = np.min(tr_bpr), vmax = np.max(tr_bpr))
			#s = f.add_subplot(243, aspect = 'equal')
			#pl.imshow(tr_bpr_patch[0,patch],vmin = np.min(tr_bpr), vmax = np.max(tr_bpr))
			#s = f.add_subplot(244, aspect = 'equal')
			#pl.imshow(tr_bpr_patch[0,patch],vmin = np.min(tr_bpr), vmax = np.max(tr_bpr))
			#s = f.add_subplot(245, aspect = 'equal')
			
			#this_patch = patches_rotated[0,int(patches_all_trials[i][0])]
			#this_patch[mask_stim] = 0
			#pl.imshow(this_patch)
			#s = f.add_subplot(246, aspect = 'equal')
			#this_patch = patches_rotated[1,int(patches_all_trials[i][0])]
			#this_patch[mask_stim] = 0
			#pl.imshow(this_patch)
			#s = f.add_subplot(247, aspect = 'equal')
			#this_patch = patches_rotated[2,int(patches_all_trials[i][0])]
			#this_patch[mask_stim] = 0
			#pl.imshow(this_patch)
			#s = f.add_subplot(248, aspect = 'equal')
			#this_patch = patches_rotated[3,int(patches_all_trials[i][0])]
			#this_patch[mask_stim] = 0
			#pl.imshow(this_patch)







		tr_bpr = np.array([bpr[np.array(patches_all_trials[:,1], dtype = int)+i] for i in range(2,5)]) #brain patches for different timepoints 
		
		for i in []:
			f = pl.figure()
			s = f.add_subplot(241, aspect = 'equal')
			tr_bpr[0,i][mask_stim] = 0
			pl.imshow(tr_bpr[0,i],vmin = np.min(tr_bpr), vmax = np.max(tr_bpr))
			s = f.add_subplot(242, aspect = 'equal')
			tr_bpr[0,i][mask_stim] = 0
			pl.imshow(tr_bpr[0,i],vmin = np.min(tr_bpr), vmax = np.max(tr_bpr))
			s = f.add_subplot(243, aspect = 'equal')
			tr_bpr[0,i][mask_stim] = 0
			pl.imshow(tr_bpr[0,i],vmin = np.min(tr_bpr), vmax = np.max(tr_bpr))
			s = f.add_subplot(244, aspect = 'equal')
			tr_bpr[0,i][mask_stim] = 0
			pl.imshow(tr_bpr[0,i],vmin = np.min(tr_bpr), vmax = np.max(tr_bpr))
			s = f.add_subplot(245, aspect = 'equal')
			patch = patches_rotated[0,np.array(patches_all_trials[:,0], dtype = int)][i]
			patch[mask_stim] = 0
			pl.imshow(patch)
			s = f.add_subplot(246, aspect = 'equal')
			patch = patches_rotated[1,np.array(patches_all_trials[:,0], dtype = int)][i]
			patch[mask_stim] = 0
			pl.imshow(patch)
			s = f.add_subplot(247, aspect = 'equal')
			patch = patches_rotated[2,np.array(patches_all_trials[:,0], dtype = int)][i]
			patch[mask_stim] = 0
			pl.imshow(patch)
			s = f.add_subplot(248, aspect = 'equal')
			patch = patches_rotated[3,np.array(patches_all_trials[:,0], dtype = int)][i]
			patch[mask_stim] = 0
			pl.imshow(patch)

		pl.show()






		

		







					
		

		
		

		
