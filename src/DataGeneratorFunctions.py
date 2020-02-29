# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:05:27 2019

These functions are used by setup.py.

There is a function that takes in the full
directory of all the files being used. It then isolates the branch so only the
directory up to the Patient number is known. Then one patient is selected to be the 
test set, then the rest of the patients are randomly distributed in to test and 
validation sets.

These patient directories are fed into another function that selects all the image
files within that directory.
Images are labeled *_1.png where * refers to the image type (Hist, ADC, T2W etc)
and the numbers after the underscore refer to which image pair it is a part of.
This function assumes the suffix is the same umbungs pairs.


- Images
    - MRI
        -Patient1
            - Class a
                -imageA_1.png
                -imageb_1.png
            - class b
                -imageA_1.png
                -imageb_1.png    
            ....
        -Patient2
            ....
        -Pateint3
        ...
        
        
    - histology
        -Patient1
            - Class a
                -histA_1.png
                -histb_1.png
            - class b
                -imageA_1.png
                -imageb_1.png    
            ....
        -Patient2
            ....
        -Pateint3
        ...


- Images
    - MRI
        -Patient1
            -Slide1
                - Class a
                    -imageA_1.png
                    -imageb_1.png
                    ....
                - class b
                    -imageA_1.png
                    -imageb_1.png 
                    .....
            -Slide2
            ....
        -Patient2
            -Slide1
                ...
            -Slide2
                ...
            ....
        -Patient3
        ...
        

In most of the examples the MRI data is first then the histology
     I.e. imageData1 is MRI imageData2 is histology

@author: lhuang
"""
import numpy as np
import glob
import os

import src.HelperFunctions as helpers
import src.ImageObject as image


#%% helper functions 

def DirCheck(Dir1, Dir2, string):
    """
    do some error checking on the two input directories.
    Inputs are lists of directories. Each directory should refer to a different group
    Checks for characters after string
    """
    for i in range(0, len(Dir1)):
        ## check if patient ids match between dataset 1 and 2
        iIndex1 = Dir1[i].rfind(string)
        sID1 = Dir1[i][iIndex1:]   
        iIndex2 = Dir2[i].rfind(string)
        sID2 = Dir2[i][iIndex2:]
        if sID2 != sID1 :
            raise ValueError('Input directory list not in the same order. Patient IDs do not match')
        # check if directories have the same number of iamges
        total1 = len(glob.glob(os.path.join(Dir1[i], '*')))
        total2 = len(glob.glob(os.path.join(Dir2[i], '*')))
        if total1 != total2:
            raise ValueError('Directory {} and {} do not have equal number of files'.format(Dir1[i], Dir2[i]))

def GetUniqueDirList(lDirs, sFind):
    """ Removed repeated directories.
    """
    sString = lDirs[0]
    iIndex = sString.rfind(sFind)
    lDirs = [item[0:(iIndex+9)] for item in lDirs]
    lUniqueDirs = np.unique(lDirs)  
    return lUniqueDirs

def GlobNFlatten(InDir, sPrefix):
    # function takes in a list of directories. Fines all files with prefix Prefix
    # in given directory. Returns a N by 1 list of all files
    Files = [glob.glob(item + sPrefix) for item in InDir]
    Files = [item for sublist in Files for item in sublist]
    return Files
         
def MaskCheck(genX1, genX2, sRootDir1, sRootDir2):
    """
    check if the image pairs in generator_two_img_from_directories are the right pair or not.
    Checks that the directory structure after sRoot1, or sRoot2 are identical
    
    Checks the entire flow generator not batch by batch
    Uses the directory structure to check if the images are in the same group.
    Looks at the last and second last directory to identify the group and classification
    """
    lFilenames1 = np.array(genX1.filenames)
    lFilenames2 = np.array(genX2.filenames)
    
    # get file names in order of the index arrays
    npFilePaths1 = lFilenames1[genX1.index_array]
    npFilePaths2 = lFilenames2[genX2.index_array]

    if not(genX1.index_array == genX2.index_array) :
        raise ValueError("indexes of pairs do not match")
    # check random transforms
    if genX1.transform_parameters != genX1.transform_parameters:
        raise ValueError('Image pairs did not have the same transforms applied')
        
    # check file names and paths
    # lFileSubPaths1 = [sDir.split(sRootDir1) for sDir in npFilePaths1[0]]    
    # lFileSubPaths2 = [sDir.split(sRootDir2) for sDir in npFilePaths2[0]]    
    # if not(lFileSubPaths1 == lFileSubPaths2) :
    #     raise ValueError("paths of the pairs do not match")
    
    # check number of images
    if genX1.n == 0 or genX2.n == 0 :
        raise ValueError("No images loaded")
    if genX1.n != genX2.n:
        raise ValueError("unequal number of input images and masks loaded")

  
def generator_img_mask(oDataGen, lDir, batch_size, seed, target_size, shuffle, color_mode, lTissueTypes, lRange = [0,1]):
    """
    Image data generator for pairs of images.
    Creates two data generators and zips them. Returns a set of images as the input and
    a set of images as the output
    
    Returns a GENERATOR functio in python, using the object oDataGen. Generators behave 
    like an iterator and iteratively
    
    https://keras.io/preprocessing/image/
    
    
    arguments:
        lDir: list of directories for image pairs   
        
    
    """
    flow_args = dict(batch_size = batch_size, 
                     target_size = target_size, 
                     seed = seed,
                     shuffle = shuffle)        
    
    # modified generators for image3, which uses a root dir and a list of patients
    sRootDir1 = os.path.split(lDir[:,0][0])[0]
    sRootDir2 = os.path.split(lDir[:,1][0])[0]
    
    # TODO: Modify to dynamicallt take in different file systems. currently lgroup list so it's igpc1008/igpc1008-slideid/class1,  igpc1008/igpc1008-slideid/class2
    lGroupList1 = list(set([os.path.join(os.path.split(sGroup)[1]) for sGroup in lDir[:,0]]))
    genX1 = oDataGen.flow_from_directory(os.path.abspath(sRootDir1), classes = [''], groups = lGroupList1, **flow_args)
    genX2 = oDataGen.flow_from_directory(os.path.abspath(sRootDir2), classes = [''], groups = lGroupList1, **flow_args)   

    # checks to ensure generators are pairing up the right image with correct mask
    MaskCheck(genX1, genX2, sRootDir1, sRootDir2)

    # seed should insure that the generators ouput the correct pairs
    trainGen = zip(genX1, genX2)
    for (img, mask) in trainGen:
        yield img[0], mask[0][:,:,:,lRange[0]:lRange[1]] # select certain channels only TODO: fix this so it works for one channel or multiple channels

def GenDataSeg(oDataGen, dir_train, dir_val, dir_test, batch_size, seed, shuffle, target_size, color_mode, lTissueTypes, lRange):
    """
    Recieves a imageoDataGenerator object. Then uses 6 sets of directories to create image data generators
    There are two pairs of training, validation and test dirs.
    """
    # data generator for testing data with no augmentations.
    TwoGenTrain = generator_img_mask(oDataGen, dir_train,  batch_size, seed, target_size, shuffle, color_mode, lTissueTypes, lRange=lRange) # a data generator object (python object)
    TwoGenVal = generator_img_mask(oDataGen, dir_val, batch_size, seed, target_size, shuffle, color_mode, lTissueTypes, lRange=lRange)
    # no augmentation, weighting or modifications on the testing generator. Also need to consider if the input image was normalized. LEave this generator out for now
    # group list is '' because lOutDir and sSubDir aleady go specifically to a subgroup
    return TwoGenTrain, TwoGenVal
               
def setupGenerators(oConfig, iNumFoldsTotal, ithFold):
    """ Sets up the image generators based on the oConfig file. Reads in arguments
    then returns 2 data generator functions. One for training and one for validation.
    """
    seed = int(oConfig['Setup']['Random Seed'])
    lImageShape = [int(sDim) for sDim in oConfig['Generator']['Image Shape'].split(',')]
    fTestSize = float(oConfig['Generator']['Test Size'])
    GroupIDPrefix = oConfig['Generator']['Group ID Prefix']
    shufflePatientOrder = oConfig['Generator'].getboolean('Patient Order Shuffle Cross Val') # shuffles order of patients list used to split into folds
    iPatientOrderSeed = seed # if seed is set, then patinet order shuffle will have a set seed
    shuffleSplitOrder = oConfig['Generator'].getboolean('Patient Order Shuffle Train Val') # shuffles the order of patients used to split into training and validation data
    lChannels = [int(x) for x in oConfig['Setup']['Channels'].split(',')]

    # put variables into a dictionary
    Gen_args = dict(batch_size = int(oConfig['Generator']['Batch Size']),
                    seed =  seed,
                    target_size = [lImageShape[0], lImageShape[1]],
                    # shuffles order of training images
                    shuffle = oConfig['Generator'].getboolean('Shuffle Train and Val Imgs'), 
                    color_mode = oConfig['Generator']['Colour Mode'],
                    lTissueTypes = oConfig['Generator']['Tissue Types'].split(','))
    

    # run data generators, and prediction generators
    oDataGen_args = dict(rescale=1./255) # add transforms to this variable so images are [0,1]
    dictAug = helpers.CreateConfigDict(oConfig, 'Augmentations') # load dict augmentations
    oDataGen_args.update(dictAug) # merge the dictionaries
    oDatagen = image.ImageDataGenerator_subdir(**oDataGen_args) # CREATE GENERATOR OBJECT
    
    #split into training, val and test sets
    npFiles = np.genfromtxt(os.path.join('..', oConfig['IO']['Input File']), delimiter=',', dtype=str) # text file that lists directories to file pairs to be read
    np1TrainDir, np1ValDir, np1TestDir = helpers.FoldNSplit(npFiles, shufflePatientOrder, iPatientOrderSeed, fTestSize, shuffleSplitOrder,  iNumFoldsTotal, ithFold)

 
    # create data generator objects for train, val and test
    DirCheck(npFiles[:,0], npFiles[:,1],  GroupIDPrefix) 
    GenTrain, GenVal = GenDataSeg(oDatagen, np1TrainDir, np1ValDir,  np1TestDir, lRange = lChannels,  **Gen_args) 
    return GenTrain, GenVal, Gen_args, np1TrainDir, np1ValDir, np1TestDir

