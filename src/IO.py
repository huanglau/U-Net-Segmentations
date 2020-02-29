
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:08:58 2019

@author: lhuang
"""

import pickle
import keras
import sys
import sklearn
import os
import glob

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shutil import copyfile
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import skimage
import skimage.io as skio
from . import ErrorMetrics as Error
import datetime
#%%

def CopyToOutDir(ithFold, iNumFoldsTotal, oConfig, sConfigFileDir, lSubDirs):
    """ sets up and copies scripts from sInDir to sOutDir. Copies the folders
    in the subdiretories in lSubDir.
    Creates a ith fold directory to output the error metrics aswell
    inputs:
        ithFold: integer. Determins which fold in a N fold cross validation
        iNumFoldsTotak: integer. total number of folds in N fold
        oConfig: configparser object. Contains paramaters in the training
    
    """
    sDirFile = oConfig['IO']['Input File']
    sInDir = os.path.dirname(os.path.abspath(__file__)) 
    oConfig['IO']['Output Dir'] = oConfig['IO']['Output Dir'] + '_' + datetime.datetime.today().strftime('%Y-%m-%d') + '_' + str(iNumFoldsTotal) + 'Folds' # update sOutDir with date

    os.makedirs(oConfig['IO']['Output Dir'],exist_ok=True)
    copyfile(sDirFile, os.path.join(oConfig['IO']['Output Dir'], os.path.split(sDirFile)[-1])) # copy list of files in .txt
    try:
        SaveScripts(sInDir, oConfig['IO']['Output Dir'], sConfigFileDir, lSubDirs)
    except:
        print("Scripts not saved properly. Likely an issue with a .py file in the configFiles directory")



def RecursiveCopy(src, sOutdir):
    """ copies all */*.py folders from src to sOutdir
    """
    os.chdir(src)
    for file in glob.glob("**/*.py", recursive = True):
        copyfile(file, os.path.join(sOutdir, file))       

def SaveScripts(sInDir, sOutDir, sConfigFileDir, lSubDirs):
    """
    Copies the scripts used in setup.py into sOutDir.
    Inputs: 
        sInDir: string. Directory being read with the code
        sOutDir: string. Path to dir where files are being copied. Error metrics will be added here 
            aswell
        sConfigFileDir: str. config file that sets up the model, loss, epochs, etc
        lSubDirs: list. list of subdirs to be made under sOutDir
            
    """
    os.makedirs(sOutDir,exist_ok=True)
    for sSubdir in lSubDirs:
        os.makedirs(os.path.join(sOutDir, sSubdir),exist_ok=True)   
    RecursiveCopy(sInDir, sOutDir)
    # copy config file
    copyfile(sConfigFileDir, os.path.join(sOutDir, 'config.ini'))
    with open(os.path.join(sOutDir, 'SoftwareVersions.txt'),"w") as file:
        file.write("Python version {} \n".format(sys.version))
        file.write("Keras version {}\n".format(keras.__version__))
        file.write("tensorflow version {}\n".format(tf.__version__))
        file.write("sklearn version {}\n".format(sklearn.__version__))
        file.write("skimage version {}\n".format(skimage.__version__))      
        file.close() 
    
def SaveTrainingHistory(sOutDir, Model, history, TrainDir, ValDir, np1TestDir):
    """
    Saves training information to sOutDir. 
    Also does some error metrics
    
    Saves ROC, accuracy and loss plots. Save the training history( val, val_acc, acc and loss)
    to a pickle file. Saves lnpPredictions and truth labels to a .txt file.
    
    """
    FigAcc, FigLoss = TrainingHistoryPlots(history)
    FigAcc.savefig(os.path.join(sOutDir,'model_accuracy.png'), dpi = 300)
    FigLoss.savefig(os.path.join(sOutDir,'model_loss.png'), dpi = 300)
    plt.close(FigAcc)
    plt.close(FigLoss)

    #save training history
    with open(os.path.join(sOutDir, 'trainHistoryDict.pkl'), 'wb') as file:
        pickle.dump([history.history], file)
        file.close()  

    with open(os.path.join(sOutDir, 'GroupSplit.txt'),"w") as file:
        file.write("\n TrainDirs \n \n")
        for item in TrainDir:
            file.write("%s\n" %item) 
        file.write("\n ValDirs\n \n ")
        for item in ValDir:
            file.write("%s\n" %item) 
        file.write("\n TestDirs\n \n")
        for item in np1TestDir:
            file.write("%s\n" %item)
        file.close()    


def SavePredImgs(save_path, npPredImgs, GenTestInput, GenTestMask, lRange, lTissueType = ['RawIM', 'TCM'], lClasses = ['nucleus','other', 'lumen'], sFileType = 'png'):
    """ Saves images, and the input and truth images
    """
    lLabels = ['TP','TN','FP','FN']
    pdConf = pd.DataFrame(columns = ['imagepath', 'fpr', 'fnr', 'auc', 'f1', 'recall', 'precision', 'Error Rate', 'optimal threshold', 'sensitivity', 'specificity', 'TN', 'TP', 'FP', 'FN'])
    
    # iterate through the prediction images
    for i, npImg in enumerate(npPredImgs):
        # get information on path, and patient id of image
        sPath = os.path.normpath(GenTestInput.filenames[i])        
        lDir = os.path.normpath(sPath).split(os.path.sep)
        lFileName = lDir[-1].split('.')

        npInput = GenTestInput.next()
        npTruth = GenTestMask.next()
        
        #TODO: THIS NEEDS TO BE GENERALIZED SO IT CAN SUIT DIFFERENT DIR PATHS STRUCCTURES
        # save predictions, truth masks and input images to output
        os.makedirs(os.path.join(save_path, 'GenImages', *lDir[-2:-1]),exist_ok=True)
        skio.imsave(os.path.join(save_path, 'GenImages', *lDir[-2:-1], lFileName[0]+'_Pred.png'), npImg )
        skio.imsave(os.path.join(save_path, 'GenImages', *lDir[-2:-1], lFileName[0]+'_Truth.png'), npTruth[0][0,:,:,:] )
        skio.imsave(os.path.join(save_path, 'GenImages', *lDir[-2:-1], lFileName[0]+'_Input.png'), npInput[0][0,:,:,:] )

        # save error maps
        npColourPredMap, lLegend =  Error.ColouredPredMap(npImg[:,:,0:1]>0.5, npTruth[0][0,:,:,0:1]>0.5)
        SaveImage(os.path.join(save_path, 'GenImages', *lDir[-2:-1]), npColourPredMap*255, lLegend, lLabels, sFileName = ('{}{}_predMap.{}'.format(lFileName[0], lClasses[0], sFileType)).replace(' ', '_'))# + lFileName[-1])
        SaveImageNoLegend(os.path.join(save_path, 'GenImages', *lDir[-2:-1]), npColourPredMap*255, sFileName = ('{}{}_predMap_NoLegend.{}'.format(lFileName[0], lClasses[0], sFileType)).replace(' ', '_'))# + lFileName[-1])
        
        # error metrics
        TN, FP, FN, TP  = Error.ConfMatrix(npTruth[0][0,:,:,0:1]>0.5, npImg[:,:,0:1]>0.5)            
        auc, fpr, tpr, thresholds = Error.GenAUC(npImg[:,:,0:1], npTruth[0][0,:,:,0:1]>0.5)
        optThresh = Error.OptimalThreshAUC(fpr, tpr, thresholds)
        FNR, FPR = Error.GenFNRFPR(Error.Thresh( npImg[:,:,0:1]), npTruth[0][0,:,:,0:1]>0.5)
        F1, recall, precision, ErrorRate = Error.GenErrorRates(Error.Thresh(npImg[:,:,0:1]),  npTruth[0][0,:,:,0:1]>0.5)
        pdConf = pdConf.append({'imagepath':sPath,
                                'fpr': FPR, 'fnr':FNR, 'auc':auc, 'f1':F1,
                                'recall': recall, 'precision':precision,
                                'Error Rate':ErrorRate,
                                'optimal threshold':optThresh,
                                'sensitivity':TP/(TP+FN),
                                'specificity':TN/(TN+FP),
                                'TP':TP,
                                'FP':FP,
                                'FN':FN,
                                'TN':TN}, ignore_index=True) 
    return pdConf

def SaveImage(sOutDir, npImage, lLegend, lLabels, sFileName = 'img.png'):
    """
    Given an image, a list of colour values, the labels for the colours output a 
    matplotlib plot. then save it to sOurDir named sFileName
    https://stackoverflow.com/questions/40662475/matplot-imshow-add-label-to-each-color-and-put-them-in-legend
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(npImage)
    patches = [mpatches.Patch(color=np.array(lLegend[i]), label=lLabels[i]) for i in range(len(lLegend)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.grid(False)
    plt.axis('off')
    plt.savefig(os.path.join(sOutDir, sFileName), bbox_inches='tight', dpi = 300, pad_inches = 0)
    plt.close()
    
def SaveImageNoLegend(sOutDir, npImage, sFileName = 'img.png'):
    """
    Given an image, the labels for the colours output a 
    matplotlib plot. then save it to sOurDir named sFileName
    https://stackoverflow.com/questions/40662475/matplot-imshow-add-label-to-each-color-and-put-them-in-legend
    """
    fig = plt.figure(frameon=False)
    w = h = 10
    fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
#    ax.imshow(int(npImage*255))
    ax.imshow(npImage)
    fig.savefig(os.path.join(sOutDir, sFileName), aspect='equal')
    plt.close()
    
    #%%
def PlotRocCurve(TruthLabels, Predictions, FlipTruthLabels = False):
    """
    Plot ROC curve
    Arguments:
            TruthLabels: Truth labels. Must be binary and a 1xN list or array
            Predictions: Predicted label. Must be binary and a 1xN list or array
            FlipTruthLabels: Switches the truth labels. Only can be used for a binary
                classification. 0 becomes 1, and 1 becomes 0. This is needed in some
                siamese network situation as the contrastive loss has 0 as identical images, 
                and 1 as the value for different images. 
                The images are usually labeled as unreg = 0, and reg = 1. So the labels need
                to be 
    """

    if FlipTruthLabels:
        TruthLabels = 1-TruthLabels
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(TruthLabels, Predictions)
    auc_keras = auc(fpr_keras, tpr_keras)
    Fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--', figure=Fig)
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras), figure=Fig)
    plt.xlabel('False positive rate', figure=Fig)
    plt.ylabel('True positive rate', figure=Fig)
    plt.title('ROC curve', figure=Fig)
    plt.legend(loc='best')
    return Fig

     
def TrainingHistoryPlots(history):
    """
    https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    Create loss, validation loss plots and accuracy and validatio accuracy plots.
    """
    Fig1 = plt.figure()
    
    # make plots for things that aren't loss
    allKeys = [str(x) for x in history.history.keys() if 'loss' not in str(x)]
    for key in allKeys:
        plt.plot(np.arange(1,len(history.history[key])+1), history.history[key], figure = Fig1)
#        plt.plot(history.history[key], figure = Fig1)
    plt.title('model accuracy', figure=Fig1)
    plt.ylabel('accuracy', figure=Fig1)
    plt.xlabel('epoch', figure=Fig1)
    plt.legend(allKeys, loc='upper left')
    # summarize history for loss
    Fig2 = plt.figure()
    plt.plot(np.arange(1,len(history.history[key])+1), history.history['loss'], figure=Fig2)
    plt.plot(np.arange(1,len(history.history[key])+1), history.history['val_loss'], figure=Fig2)
    plt.title('model loss', figure=Fig2)
    plt.ylabel('loss', figure=Fig2)
    plt.xlabel('epoch', figure=Fig2)
    plt.legend(['train', 'val'], loc='upper left')
    return Fig1, Fig2

    