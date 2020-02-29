# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:32:19 2019
Set up to run the setup_segmentation folder from the command line by reading
a oConfig.ini file to set the parameters along with arguments in the command line

The only arguments in the command line are the integer that tells us which batch the 
validation is on, and the total number of batches


This code only does segmentation. Uses U-Net.
DEBUG GUIDE:
    If the classifier isn't running well, here are some list of things that might be the issue
        - Check the input images. If the program needs to appy augmentation to the image 
            on the fly, then it decreases the GPU usage, and increases training time. To easily
            increase GPU usage, consider adjusting the size of the input images on the harddrive so it matches the image
            size the network expects. This will decrease the amount of work the cpu needs to do before sending it to the gpu. 
        - Not enough epochs. Consider increasing number of epochs and increase training time
        - if accuracy is doesn't make sense, look at the datagenerator to make sure it's
              outputting the right images, and the right classification
        - check that steps per epoch is > 0. This might cause a bug in the code
        - check specific file path of the input images
        - check that the labels don't have spaces in them. Make sure they match the subdir structure
        - check learning rate. Maybe adjust it lower or higher. Decreasing the learning rate often improves 
            the classifier if it's overfitting
        - check if the classes are balanced. In a segmentation problem this might mean
            taking into consideration how much of one class is present in an image on average.
            You may consider duplicating certain training data. Make sure you don't include these
            duplications in the validation or testing data.
        - check if the loss function is a good one
            - Since this is a segmentation problem, duplicating input data may not work. In this case
                you may consider using a weighted loss. You may try a loss that has 
                a weight to penalize false positives more than false negatives. You may also
                consider where the errors occur relative to the segmentation. For example you 
                may want to penalize incorrect pixel classifications at the edge of a object more than
                incorrect pixel classifications in the center of an object.
                This project includes a dynamically weighted binary cross entropy and a fixed
                weighted bianry cross entropy you might consider.
        - check if input training images are normalized to [0,1] and if the val or test data is also normalized [0,1]
        - adjust network.
            - add drop out layers or remove drop out layers. Add regularization to the layers
        - check if the train and test and val dirs have the same balance of classes. Depending on how the 
            training data is split, you may have ended up with an unbalanced set of training data but a balanced
            set of testing data.
        
    If the training accuracy is much higher than the validation accuracy then the model is likely overfitting.
        expectially if the val accuracy is not changing, but train accuracy is increasing
        - add droppout layers to the network
        - add some augmentation
        - decrease learning rate
        - increase datasize
        - apply early stopping
        - adjust loss function 
        - add regulrization. 'https://keras.io/regularizers/' These area applied on a per layer basis



@author: lhuang
"""

import configparser
import os
import sys
import keras
import keras.backend as K

import src.DataGeneratorFunctions as DataGenFun
import models.uNet as uNet
import src.Loss as loss
import src.IO as IO
import src.TrainTest as train
from contextlib import redirect_stdout


def setModelArgs(oConfig):
    """

    Parameters
    ----------
    oConfig : configini object
        config object that specifies training params along with where training data
        is stored

    Returns
    -------
    Model_args : Dictionary
        dictionary contaning arguments used for model building
    
    sets up model arguments with oConfig.
    Sets loss, and error metrics

    """

    Model_args = dict()
    # select the loss,
    if oConfig['Model']['Loss'] == 'binary_crossentropy':
        Model_args['loss'] = 'binary_crossentropy'
    elif oConfig['Model']['Loss'] == 'weighted_categorical_crossentropy':
        Model_args['loss'] = loss.weighted_categorical_crossentropy
    elif oConfig['Model']['Loss'] == 'dyn_weighted_binary_crossentropy':
        Model_args['loss'] = loss.dyn_weighted_binary_crossentropy
    # select optimizer
    if oConfig['Model']['Optimizer'] == 'adam':
        Learning_args = dict(lr=float(oConfig['Model']['Learning Rate'])) 
        # TODO: adjust so model optimizers takes in a dict. Currently this is not well tested
        if oConfig.has_option('Model', 'Decay'):
            Learning_args['decay'] = float(oConfig['Model']['Decay'])
        if oConfig.has_option('Model', 'momentum'):
            Learning_args['momentum'] = float(oConfig['Model']['momentum'])
        if oConfig.has_option('Model', 'amsgrad'):
            Learning_args['amsgrad'] = oConfig['Model']['amsgrad'] == 'True'
        else:
            Model_args['optimizer'] = keras.optimizers.Adam(**Learning_args)
    
    # select the loss functions.
    # TODO: adjust so loss functions can be selected in the config file
    Model_args['metrics'] = ['accuracy', loss.Recall, loss.Precision]
    return Model_args

def CompileModel(oConfig,  iNumFoldsTotal):
    """
    
    Parameters
    ----------
    oConfig : configini object
        config object that specifies training params along with where training data
        is stored
    iNumFoldsTotal : int
        number of folds in this experiment

    Returns
    -------
    NewModel : keras model object
        Compiled model to be used for training

    """
    
    # read in model arguments form config object
    Model_args = setModelArgs(oConfig)
    
    #initialize model
    lImageShape = [int(sDim) for sDim in oConfig['Generator']['Image Shape'].split(',')]
    NewModel = uNet.unet(lInputSize = lImageShape)
    NewModel.summary() 
    
    # print model summary to a file
    with open(os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', oConfig['IO']['Output Dir'],'modelsummary.txt'), 'w+') as file:
        with redirect_stdout(file):
            NewModel.summary()
    
    NewModel.compile(**Model_args)
    return NewModel

#%%
def main(sConfigFileDir, ithFold, iNumFoldsTotal):
    """
    

    Parameters
    ----------
    sConfigFileDir : string
        path to config.ini file that specifies training and testing params
    ithFold : int
        indicates which fold this experiment is on
    iNumFoldsTotal : int
        indicates total number of folds in this experiment

    Raises
    ------
    ValueError
        If no config.ini file is found.

    Returns
    -------
    pdResults : dataframe
        contains error metrics from experiment

    """
    if ithFold >= iNumFoldsTotal:
        raise ValueError('cannot do {}th fold when there are only {} folds in total'.format(ithFold, iNumFoldsTotal))
    if os.path.exists(sConfigFileDir) != True:
        raise ValueError('{} not found'.format(sConfigFileDir))
    
    # set up config files
    oConfig = configparser.ConfigParser()
    oConfig.read(sConfigFileDir)
    
    # copy code, and config file to new output directory
    #TODO: replace the lSubDir with a function that just recursively checks for all dirs
    IO.CopyToOutDir(ithFold, iNumFoldsTotal, oConfig, sConfigFileDir,
                 lSubDirs = ['src', 'tests', 'tests/ErrorMetrics', 'models'])
    
    # set up image generator 
    GenTrain, GenVal, Gen_args, np1TrainDir, np1ValDir, np1TestDir = DataGenFun.setupGenerators(oConfig, iNumFoldsTotal, ithFold)
    
    # set up model
    NewModel = CompileModel(oConfig, iNumFoldsTotal)
    print('Model Created \n{:.0f}th fold \n Data generated with {:.0f} folds in total'.format(ithFold+1, iNumFoldsTotal))
    
    # train data and do error metricsi
    pdResults = train.TrainAndTestUNet(oConfig, iNumFoldsTotal, ithFold, NewModel,  GenTrain = GenTrain, GenVal = GenVal, np1TrainDir = np1TrainDir, np1ValDir = np1ValDir, np1TestDir = np1TestDir) 
    K.clear_session()
    return pdResults

# if you are doing N fold cross validation it is better to call python from the command line
# this is so that the keras 'session' is fully reset each fold, and GPU memory is fully released
# There are methods to clear keras in the script but I"m usure if it's been debugged properly yet
if __name__ == "__main__":
    """ (classes, sDirList, iSeed, sOutDir)
    """  
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    

# uncomment this if you are running an experiment not in command line
# if you want to run a 1fold experiment  use the following line
# main('config_segmentation.ini', 0, 1)
