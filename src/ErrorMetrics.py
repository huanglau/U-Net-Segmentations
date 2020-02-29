# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:39:02 2019
Error metric functions


@author: lhuang
"""
import os
from sklearn import metrics
import pandas as pd
import numpy as np 
from . import IO as IO
import src.ImageObject as image
#%% error metric functions

def Thresh(npImage, fThreshVal = 0.5):
    """ Thresholds a 2D multiple channel image
    """
    npThreshImage = np.zeros(np.shape(npImage))
    npThreshImage[npImage >= fThreshVal] = 1
    npThreshImage[npImage < fThreshVal] = 0
    return npThreshImage

def ConfMatrix(npTruthImg, npPredImg):
    """ uses parallel computing to generate confusion matrix 
    Found to be faster than sklearn's confusion_matrix for large images at least
    
    Can be a list of images.
    
    Returns  [tn, fp, fn, tp] ints
    """
    npPredImg = npPredImg.astype(np.bool)
    npTruthImg  = npTruthImg.astype(np.bool)
    npTP = np.logical_and(npPredImg, npTruthImg)
    npTN = np.logical_and(np.invert(npPredImg), np.invert(npTruthImg))
    npFN = np.logical_and(np.invert(npPredImg), npTruthImg)     
    npFP = np.logical_and(npPredImg, np.invert(npTruthImg))    
    #TODO: check if npTP is a numpy array
    return [np.sum(npTN), np.sum(npFP), np.sum(npFN), np.sum(npTP)]#confusion_matrix(npTruthImg.flatten(), npPredImg.flatten(), labels = lLabels).ravel()  


def ConfMatrixFromErrorMap(npErrorMap, lColours):
    """ uses GPU to generate confusion matrix 
    Uses an error map that is labeled, with colours lColours
    
    Counts the occurence of every colour on the list. This can be used
    as a count for TP, TN, FP, FN if the colours in the list correspond to those 
    categories
    
    """
    lCounts = [np.sum(np.all(npErrorMap == np.array(lColour), axis=2)) for lColour in lColours]
    if np.sum(lCounts) != np.prod(np.shape(npErrorMap)[:2]):
        raise ValueError('Error: more prediction labels than pixels in image')
    return lCounts

#%% error metrics for classifications
def GenAUC(npPred, npTruth, iPosLabel = 1):
    """
    generates auc, false pos rate, true pos rate and thresholds of each given a prediction and
    truth numpt array. Should work in any dimentional data. 
    Assumes binary classification and that a positive result is a 1
    """
    if (np.sum(npTruth==1) + np.sum(npTruth==0)) != np.shape(npTruth.flatten()):
        raise ValueError('AUC failed. Truth map is not binary.')
    fpr, tpr, thresholds = metrics.roc_curve(npTruth.flatten(), npPred.flatten(), pos_label = iPosLabel)
    return metrics.auc(fpr, tpr), fpr, tpr, thresholds

def OptimalThreshAUC(fpr, tpr, thresholds):
    """ returns the optimal threshold value in a binary classification
    when using an ROC caluclator
    
    Chose the optimal threshold by finding the point on the ROC curve that has the minimal 
    distance from (0,1). This was calculated by simple geometry. Use pythagras therom to find the 
    shortest distance from 1. The x-axis is fpr. Let the fpr for a given threshold be b.
    The y-axis is tpr. Let the tpr for a given threshold be a. The distance from a threshold
    to the point (0,1) is np.sqrt(fpr**2+(1-tpr)**2)
    """
    distance = np.sqrt((1-tpr)**2+fpr**2)
    index_min = np.argmin(distance)
    return thresholds[index_min]
    
def GenErrorRates(npPred, npTruth):
    """ gets a set of data, finds the F1,recall, precision, and error rate 
    Assumes prediction is already thresholded.
    
    Only works for boolean classifications
    
    REturns 
    F1,recall, precision, and error rate 
    """
    if np.shape(npPred) != np.shape(npTruth):
        raise ValueError('prediction and truth labels must be the same size')
    if np.sum((npTruth !=0) * (npTruth != 1.0)) > 0 or np.sum((npPred !=0) * (npPred != 1.0)) > 0 :
        raise ValueError('inputs must be 0s or 1s')
    npTN, npFP, npFN, npTP = ConfMatrix(npTruth, npPred)
    recall = npTP/(npTP+npFN)
    precision = npTP/(npTP+npFP)
    ErrorRate = (npFP+npFN)/(npTN+npFP+npFN+npTP)
    F1 = 2*precision*recall/(precision+recall)
    
    return F1, recall, precision, ErrorRate 

def GenFNRFPR(npPred, npTruth):
    """ gets a set of data, finds the false negative rate and false positive rate
    
    """
    if np.shape(npPred) != np.shape(npTruth):
        raise ValueError('prediction and truth labels must be the same size')
    if np.sum((npTruth !=0) * (npTruth != 1.0)) > 0 or np.sum((npPred !=0) * (npPred != 1.0)) > 0 :
        raise ValueError('inputs must be 0s or 1s')
    npTN, npFP, npFN, npTP = ConfMatrix(npTruth, npPred) # .sum(npTN), np.sum(npFP), np.sum(npFN), np.sum(npTP)
    FNR = npFN/(npFN+npTP)
    FPR = npFP/(npFP+npTN)
    return FNR, FPR
 
def CalcErrorRates(npResults, npTruthValues, pdConf, sPatID, sSlideID, lClasses):
    """ Calculates AUC, fpr, tpr, thresholds optimal thresholds, for npresults and npvalues
    returns an pdConf that has all the values added
    """
    auc, fpr, tpr, thresholds = GenAUC(npResults, npTruthValues)
    optThresh = OptimalThreshAUC(fpr, tpr, thresholds)
    F1, recall, precision, ErrorRate = GenErrorRates(Thresh(npResults[:,0]), npTruthValues)
    pdConf = pdConf.append({'PatID':sPatID, 'Slide ID':sSlideID,
                                'fpr': fpr, 'tpr':tpr, 'auc':auc, 'f1':F1,
                                'recall': recall, 'precision':precision,
                                'Error Rate':ErrorRate,
                                'optimal threshold':optThresh}, ignore_index=True)  

def ColouredPredMap(npPredict, npTruth):     
    """ 
    Generates a fp,fn,tn, map given a set of images
    Assumes images are binary, 
    
    CURRENTLY ONLY TESTED FOR SINGLE CHANNEL PREDICTIONS
    """
    # create colour legend
    white = [1,1,1] # true pos
    black = [0, 0, 0]# true neg
    red = [1, 0, 0]  # false pos
    blue = [0, 0, 1]  # false neg
    npPredict = npPredict.astype(np.bool)
    npTruth  = npTruth.astype(np.bool)
    npPredictionMap = np.zeros(shape=(npPredict.shape[0], npPredict.shape[1], 3), dtype=np.uint8)
    npTP = np.logical_and(npPredict, npTruth)
    npTN = np.logical_and(np.invert(npPredict), np.invert(npTruth))
    npFN = np.logical_and(np.invert(npPredict), npTruth)     
    npFP = np.logical_and(npPredict, np.invert(npTruth))            
    npPredictionMap[npTP[:,:,0],:] = white
    npPredictionMap[npTN[:,:,0],:] = black
    npPredictionMap[npFP[:,:,0],:] = red
    npPredictionMap[npFN[:,:,0],:] = blue
    lLegend = [black, red, blue, white] #  tn, fp, fn, tp
    if np.sum(npTP) + np.sum(npTN) + np.sum(npFN) + np.sum(npFP) != npPredict.shape[0] *npPredict.shape[1] :
        raise ValueError('Number of labeled pixels does not equal the number of pixels in the image')
    return npPredictionMap, lLegend

#%%



def PredictionErrorPerClass(sOutDir, npTestDir, Model, iImageSize, lRange, NewSize = [240,240], sReportName = 'ErrorReport.txt', iNumSave = 7, iNumPredict = None, fThreshVal = 0.5):
    """ runs a prediction generator on the model given a numpy array of testing directories
    Runs a error metric functino for segmentation
    Generates the prediction images, then saves iNumSave of them to sOutDir
    Can generate for all images in the testing directory, or onlya select number of images
    
    Uses the data generator to generte the expected masks
    
    This assumes the testing data is in a very specific file directory path. specified at top of file
    It expects a directory system to be something like 
    
        
    Arguments: 
        sOutDir: directory where the resulting prediction maps and error reports will be placed
        npTestDir: numpy array containing group IDs of the input and output images
                    [[./MRI/group1/, ./hist/group1/],[./MRI/group2/, ./hist/group2/],[./MRI/group3/, ./hist/group3/]]
        Model: keras model object
        iImageSize: size of images for network
        lRange: indicates which channel from the expected images to look at
        sReportName: = 'ErrorReport.txt'
        iNumSave = 7
        iNumPredict = None
        fThreshVal = 0.5

    """
    if type(npTestDir) != type(np.array([])):
        raise ValueError ('testing directory much be a list')
    dfError = pd.DataFrame(columns = ['imagepath', 'fpr', 'fnr', 'auc', 'f1', 'recall', 'precision', 'Error Rate', 'optimal threshold', 'sensitivity', 'specificity', 'TN', 'TP', 'FP', 'FN'])

    # split up into patients
    for i, sPatDir in enumerate(npTestDir) :
        
        # create a data generator without augmentation
        TestDataGen = image.ImageDataGenerator_subdir()
        flow_args = dict(batch_size = 1, 
                         target_size = [iImageSize[0], iImageSize[1]],
                         seed = 0,
                         shuffle = False,
                         classes = [''])   
        
        TruthInputGen = TestDataGen.flow_from_directory(sPatDir[0], groups = [''], **flow_args) # truth values
        TruthMaskGen = TestDataGen.flow_from_directory(sPatDir[1], groups = [''], **flow_args) # truth masks
        
        if TruthInputGen.n > 0 :
            # check for if the num predictions is greater than the actual num images
            if iNumPredict == None or iNumPredict > TruthInputGen.n :
                npResults = Model.predict_generator(TruthInputGen, steps = TruthInputGen.n, verbose=1)
            else: 
                npResults = Model.predict_generator(TruthInputGen, steps = iNumPredict, verbose=1)
                
            # save prediction images and generate error metrics
            if iNumSave > TruthInputGen.n:
                pdConf = IO.SavePredImgs(sOutDir, npResults, TruthInputGen, TruthMaskGen, lRange = lRange) # save 30 samples
            else:
                pdConf = IO.SavePredImgs(sOutDir, npResults[:iNumSave,:,:,:], TruthInputGen, TruthMaskGen, lRange = lRange)
            dfError = dfError.append(pdConf) 

    dfError.to_csv(os.path.join(sOutDir, 'ErrorMetrics.csv'))                    
    return dfError
