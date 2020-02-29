# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:41:31 2019

Holds the training and testing functions for a CNN

@author: lhuang
"""
import pickle
import math
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
import src.IO as IO
import src.ErrorMetrics as Error

def TrainModel(sOutDir, Model, GenTrain, GenVal, Fit_args, np1TrainDir, np1ValDir, np1TestDir, kFold):
    """
    Trains Model using GenTrain and GenVal data. These are imagedatagenerators
    Uses TestData, a numpy array, as the prediction values
    """
    # paramaters for an early stop, and also saves the model after each epoch overwriting
    if not os.path.exists(os.path.join(sOutDir,str(kFold) +'thFold')):    
        os.makedirs(os.path.join(sOutDir,str(kFold) +'thFold'),exist_ok=True) 
    earlystop =[ ModelCheckpoint(os.path.join(sOutDir,str(kFold) +'thFold', 'trainedModel.h5'), monitor='val_acc')]

    # fit model
    history = Model.fit_generator(GenTrain,
                               validation_data = GenVal,
                               callbacks = earlystop,
                               **Fit_args) 
    
    # saves unet, and some training plots
    IO.SaveTrainingHistory(os.path.join(sOutDir, str(kFold) + 'thFold'), Model, history, np1TrainDir, np1ValDir, np1TestDir)
    with open(os.path.join(sOutDir, str(kFold) + 'thFold', 'trainHistoryDict.pkl'), 'wb') as file:
        pickle.dump([history.history], file)
        file.close()
    return Model

def TrainAndTestUNet(oConfig, iNumFoldsTotal, kFold, Model, GenTrain, GenVal, np1TrainDir, np1ValDir, np1TestDir, iNumSave = 20):
    """
    Trains Model using GenTrain and GenVal data. These are imagedatagenerators
    Uses TestData, a numpy array, as the prediction values
    """
    # set up training parameters
    iNumSave = int(oConfig['IO']['Number Save']) # number of prediciton maps saved per group
    lImageSize = [int(sDim) for sDim in oConfig['Generator']['Image Shape'].split(',')]
    lRange = [int(x) for x in oConfig['Setup']['Channels'].split(',')]
    npFiles = np.genfromtxt(os.path.join('..', oConfig['IO']['Input File']), delimiter=',', dtype=str) # text file that lists directories to file pairs to be read
    # add escape to suuper directory because this file is in .src
    sOutDir = os.path.join('..', oConfig['IO']['Output Dir'])
    iApproxNumImages =  sum([len(files) for directory in npFiles[:,0] for r, d, files in os.walk(os.path.join(os.path.dirname(__file__), directory))])

    if iApproxNumImages < 1:
        raise ValueError('no images found')
        
    # this fold factor is used to help determine the steps per epoch
    # this condition is added so that in the case of only 1 fold, this factor will not
    # become 0 and break the code.
    fFoldFactor = (iNumFoldsTotal-1)/iNumFoldsTotal
    if fFoldFactor < 1 :
        fFoldFactor = 1
        
    # can add tramsforms to the Fit_args        
    if oConfig['Generator']['Steps per epoch'] == 'Auto':
        fTestSize = float(oConfig['Generator']['Test Size'])
        Fit_args = dict(steps_per_epoch = math.ceil(iApproxNumImages//int(oConfig['Generator']['Batch Size'])*(1-fTestSize)*fFoldFactor),
                          epochs = int(oConfig['Model']['Epochs']),
                          validation_steps = math.ceil(iApproxNumImages//int(oConfig['Generator']['Batch Size'])*fTestSize * fFoldFactor))
    else :
        Fit_args = dict(steps_per_epoch = int(oConfig['Generator']['Steps per epoch']),
                          epochs = int(oConfig['Model']['Epochs']),
                          validation_steps = int(oConfig['Generator']['Steps per epoch']))
        
    # check if error occured in step size calculations
    if Fit_args['validation_steps'] < 1 or Fit_args['steps_per_epoch'] < 1:
        raise ValueError('validation step size or training step size less than 1. this is an invalid input')
        
    # train model
    Model = TrainModel(sOutDir, Model, GenTrain, GenVal, Fit_args, np1TrainDir, np1ValDir, np1TestDir, kFold)
    
    #TODO: add search for iNumPred in oConfig object
    iNumPredict = oConfig['IO']['Number Predict']
    if iNumPredict == 'All':
        # if iNumPredict is none, then it will predict on the entire dataset
        iNumPredict = None
    else:
        iNumPredict = int(iNumPredict)

    Error.PredictionErrorPerClass(os.path.join(sOutDir, str(kFold) + 'thFold'), np1TestDir, Model, lImageSize,  iNumSave = iNumSave, lRange = lRange, iNumPredict = iNumPredict)
