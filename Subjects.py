#!/usr/bin/env python
# encoding: utf-8
"""
analyze_7T_S1.py

Created by Tomas HJ Knapen on 2009-11-26.
Modified by Dirk van Moorselaar on 2014-05-09.
Copyright (c) 2009 TK. All rights reserved.
"""

import os, sys, datetime
import subprocess, logging

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

this_raw_folder = '/home/raw_data/WM_patch/'
this_project_folder = '/home/shared/WMM_PRF/'

sys.path.append(os.environ['ANALYSIS_HOME'])

from Tools.Sessions import *
from Tools.Subjects.Subject import *
from Tools.Run import *
from Tools.Projects.Project import *

from WMMappingSession import *

which_subject = 'TK'

def runWholeSession( rA, session ):
	for r in rA:
#		if r['scanType'] == 'epi_bold':
#			r.update(epiRunParameters)
		thisRun = Run( **r )
		session.addRun(thisRun)
	session.parcelateConditions()
	session.parallelize = True

	### Creating folder hierarchy
	
	#session.setupFiles(rawBase = presentSubject.initials, process_eyelink_file = False)
	

	### Preprocessing functional data
	
	# check whether the inplane_anat has a t2 or t1 - like contrast. t2 is standard. else add contrast = 't1'
	#session.registerSession() # deskull = False, bb = True, flirt = True, MNI = True
	
	# after registration of the entire run comes motion correction
	
	#session.motionCorrectFunctionals()
	#session.resample_epis(conditions=['WMM'])
	# 
	# CHECK REGRISTRATION HERE!!
	
	#session.createMasksFromFreeSurferLabels(annot = False, annotFile = 'aparc.a2009s', labelFolders = ['for_remapping']) # cortex = True, why? not supported argument
	#session.create_dilated_cortical_mask(dilation_sd = 0.25, label = 'cortex') # returns error 
	#  
	#session.rescaleFunctionals(operations = ['sgtf'], filterFreqs = {'highpass':48, 'lowpass':0}, funcPostFix = ['mcf'])
	#session.rescaleFunctionals(operations = ['percentsignalchange'], filterFreqs = {'highpass':48, 'lowpass': 0}, funcPostFix = ['mcf', 'sgtf'])
	 
	### Behavior, eye, physio

	#session.stimulus_response_timings()
	#session.stimulus_timings_unique_PRF()
	#session.physio()
	#session.eye_timings(nr_dummy_scans = 6, mystery_threshold = 0.05,saccade_duration_threshold = 10)


	### Run GLM

	session.runAllGLMS()
	#session.setupRegistrationForFeat()
	# gfeat is done maually in fsl. this happens in mni space. now we want to register our gfeats to our functional space:
	# session.gfeat_analysis()


	### Store preprocessed data in hdf5 file
	#session.remove_mask_stats()
	#session.mask_stats_to_hdf(sj_PRF_Session_folder = sj_PRF_Session_folder)

	### First check PRF
	#session.compare_stim_input()
	
	#session.timings_across_runs()
	#session.compare_brain_and_stimulus_input_data()


	### Deconvolution
	#session.deconvolve_roi(roi = ['V1'], threshold = 10.0)	
	#session.deconvolve_roi(roi = ['V2'], threshold = 10.0)	
	#session.deconvolve_roi(roi = ['MT'], threshold = 10.0)	

	#session.decode_patches_from_roi()

	# session.rescale_images()
	# session.setup_all_data_for_predict(['V1'])
	# session.predict_patches(roi = ['V1'], contrast = 'zstat', nr_of_elements = 60, masked = True)
	# session.zscore_timecourse_per_condition()
	
	# session.design_matrix()
	# session.fit_PRF(n_pixel_elements = 40, mask_file_name = 'cortex', n_jobs = -2)

	# session.GLM_for_nuisances()
	# session.fit_PRF(n_pixel_elements = 42, mask_file_name = 'cortex_dilated_mask', n_jobs = 28)
	# session.results_to_surface(res_name = 'corrs_cortex_dilated_mask')
	# session.RF_fit(mask_file = 'cortex_dilated_mask', stat_threshold = -50.0, n_jobs = 28, run_fits = False)
	# session.results_to_surface(res_name = 'ecc', frames = {'rad':0})
	# session.results_to_surface(res_name = 'polar', frames = {'rad':0})
	# session.makeTiffsFromCondition(condition = 'PRF', y_rotation = 90.0, exit_when_ready = 0)

# for testing;
if __name__ == '__main__':
	if which_subject == 'MF':
		# subject information
		initials = 'MF'
		firstName = 'Michel'
		standardFSID = 'MF_200813_12'
		birthdate = datetime.date(1986, 10, 29)
		labelFolderOfPreference = ''
		presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
	
		presentProject = Project( 'WMM_PRF', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
	
		sessionDate = datetime.date(2014, 1, 22)
		sessionID = 'WMM_PRF' + presentSubject.initials
		sj_init_data_code = 'MF_200414'
	
		WMM_Session = WMMappingSession(sessionID, sessionDate, presentProject, presentSubject)
	
		try:
			os.mkdir(os.path.join(this_project_folder, 'data', initials))
			os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
		except OSError:
			WMM_Session.logger.debug('output folders already exist')
	
	
		WMM_run_array = [
			{'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + sj_init_data_code +  '/raw/mri/', 'Michel_Failing_WIP_T2W_RetMap_1.25_CLEAR_5_1.nii.gz' ), 
				},
			{'ID' : 3, 'scanType': 'epi_bold', 'condition': 'WMM', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + sj_init_data_code +  '/raw/mri/', 'Michel_Failing_WIP_RetMap_2.5_1.5_SENSE_3_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_project_folder, 'raw_data/', 'mf_1.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + sj_init_data_code +  '/raw/behavior/', 'mf_1_2014-04-20_12.26.20_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/hr/', 'MF_RetMap1.log' ), 
				},
			{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'WMM', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/mri/', 'Michel_Failing_WIP_RetMap_2.5_1.5_SENSE_4_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_project_folder, 'raw_data/', 'mf_2.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/behavior/', 'mf_2_2014-04-20_12.41.49_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/'   + sj_init_data_code +  '/raw/hr/', 'MF_RetMap2.log' ), 
				},
			{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'WMM', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/mri/', 'Michel_Failing_WIP_RetMap_2.5_1.5_SENSE_6_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_project_folder, 'raw_data/', 'mf_3.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/behavior/', 'mf_3_2014-04-20_13.09.44_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/'   + sj_init_data_code +  '/raw/hr/', 'MF_RetMap3.log' ), 
				},
			{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'WMM', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/mri/', 'Michel_Failing_WIP_RetMap_2.5_1.5_SENSE_7_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_project_folder, 'raw_data/', 'mf_4.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/behavior/', 'mf_4_2014-04-20_13.26.00_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/'   + sj_init_data_code +  '/raw/hr/', 'MF_RetMap4.log' ), 
				},
		]
	
		runWholeSession(WMM_run_array, WMM_Session)
	
	elif which_subject == 'TK':
		# subject information, 2nd subject
		initials = 'TK'
		firstName = 'Tomas'
		standardFSID = 'TK_091009tk' 
		birthdate = datetime.date( 1985, 04, 05 )
		labelFolderOfPreference = ''
		presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
		
		presentProject = Project('WMM_PRF', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
		
		sessionDate = datetime.date(2014, 04, 20)
		sessionID = 'WMM_PRF' + presentSubject.initials
		sj_init_data_code = 'TK_200414'
		
		sj_PRF_Session_folder = '/home/shared/PRF/data/TK/TK_130813'

		WMM_Session = WMMappingSession(sessionID, sessionDate, presentProject, presentSubject)
		
		try:
			os.mkdir(os.path.join(this_project_folder, 'data', initials))
			os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
		except OSError:
			WMM_Session.logger.debug('output folders already exist')
			
			
		WMM_run_array = [
			{'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
				'rawDataFilePath': os.path.join(this_project_folder, 'raw_data/' + sj_init_data_code +  '/raw/mri/', 'TK_WIP_T2W_RetMap_1.25_CLEAR_5_1_Flip.nii.gz' ), 
				},
			{'ID' : 3, 'scanType': 'epi_bold', 'condition': 'WMM', 
				'rawDataFilePath': os.path.join(this_project_folder, 'raw_data/' + sj_init_data_code +  '/raw/mri/', 'TK_WIP_RetMap_2.5_1.5_SENSE_3_1_Flip.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_project_folder, 'raw_data/', 'TK_1.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + sj_init_data_code +  '/raw/behavior/', 'tk_1_2014-04-20_15.29.49_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/hr/', 'TK_RetMap1.log' ), 
				},
			{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'WMM', 
				'rawDataFilePath': os.path.join(this_project_folder, 'raw_data/'  + sj_init_data_code +  '/raw/mri/', 'TK_WIP_RetMap_2.5_1.5_SENSE_4_1_Flip.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_project_folder, 'raw_data/', 'TK_2.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/behavior/', 'tk_2_2014-04-20_15.44.16_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/'   + sj_init_data_code +  '/raw/hr/', 'TK_RetMap2.log' ), 
				},
			{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'WMM', 
				'rawDataFilePath': os.path.join(this_project_folder, 'raw_data/'  + sj_init_data_code +  '/raw/mri/', 'TK_WIP_RetMap_2.5_1.5_SENSE_8_1_Flip.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_project_folder, 'raw_data/', 'TK_5.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/behavior/', 'tk_3_2014-04-20_16.33.56_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/'   + sj_init_data_code +  '/raw/hr/', 'TK_RetMap5.log' ), 
				},
		]
	
		runWholeSession(WMM_run_array, WMM_Session)

	elif which_subject == 'DM':
		# subject information, 2nd subject
		initials = 'DM'
		firstName = 'Dirk'
		standardFSID = 'DM_220813_12' 
		birthdate = datetime.date( 1983, 06, 03 )
		labelFolderOfPreference = ''
		presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
		
		presentProject = Project('WMM_PRF', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
		
		sessionDate = datetime.date(2014, 1, 22)
		sessionID = 'WMM_PRF' + presentSubject.initials
		sj_init_data_code = 'DM_200414'
		
		WMM_Session = WMMappingSession(sessionID, sessionDate, presentProject, presentSubject)
		
		try:
			os.mkdir(os.path.join(this_project_folder, 'data', initials))
			os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
		except OSError:
			WMM_Session.logger.debug('output folders already exist')
			
			
		WMM_run_array = [
			{'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + sj_init_data_code +  '/raw/mri/', 'Dirk_M_WIP_T2W_RetMap_1.25_CLEAR_5_1.nii.gz' ), 
				},
			{'ID' : 3, 'scanType': 'epi_bold', 'condition': 'WMM', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/' + sj_init_data_code +  '/raw/mri/', 'Dirk_M_WIP_RetMap_2.5_1.5_SENSE_3_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_project_folder, 'raw_data/', 'dm_1.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/' + sj_init_data_code +  '/raw/behavior/', 'dm_1_2014-04-20_13.48.19_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/hr/', 'DM_RetMap1.log' ), 
				},
			{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'WMM', 
				'rawDataFilePath': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/mri/', 'Dirk_M_WIP_RetMap_2.5_1.5_SENSE_4_1.nii.gz' ), 
				'eyeLinkFilePath': os.path.join(this_project_folder, 'raw_data/', 'dm_2.edf' ), 
				'rawBehaviorFile': os.path.join(this_raw_folder, 'data/'  + sj_init_data_code +  '/raw/behavior/', 'dm_2_2014-04-20_14.04.25_outputDict.pickle' ), 
				'physiologyFile': os.path.join(this_raw_folder, 'data/'   + sj_init_data_code +  '/raw/hr/', 'DM_RetMap2.log' ), 
				},
		]
	
		runWholeSession(WMM_run_array, WMM_Session)

