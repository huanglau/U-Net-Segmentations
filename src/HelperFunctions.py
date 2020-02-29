# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:01:59 2019

These helpers are associated with saving the model, training, testing and saving 
accuracies, plots and roc curves


@author: lhuang
"""

import numpy as np
import sklearn
import random
import sklearn.model_selection

#%%


def SplitDirs(npFiles, np2FoldIndexes, fTestSize, iFold,  bShuffle = True, iSeed = 0):
    """ splits a numpy array depending on a numpy array. Removes the ith index, then splits the rest
    of the list into training, validation, and testing sets
    
    Arguments:
        npFiles: list of file pairs. This list will be split
        np2FoldIndexes: list of indexes to be split in the folds, Shape 2xN
        iFold: ith fold
        fTestSize: size of the testing batch. 
        bShuffle: wheather or not the training and validation list will be shuffled
        iSeed: random seed for the shuffle
        
    returns:
        np1TrainDir, np1ValDir, np1TestDir: npFiles but split into 3 groups
            depending on n3FoldIndexes.
            From np2FoldIndexes the iFold group of indexes is selected. The remaining
            indexes in np2FoldIndexes is shuffled or not, then split into training and validation.
    
    example:
        npFiles = [g1, g2, g3, g4, g5, g6]
        np2FoldIndexes = [[0,1],[2,3],[4,5]]
        iFold = 0
        fTestSize = 0.25, shufffle = True
        Returns: npTestDir = [g1, g2],  npValDir = [np4], npTrainDir = [g6, g3, g5]

    """
    np1TestDir = npFiles[np2FoldIndexes[iFold]]
    # remove all directories but ith one
    lUniqueDirNoI = npFiles[FlattenAndRemoveIndex(np2FoldIndexes, iFold)]
    # randomly split the remaining files (train and val) by splitting an index
    TrainIndex, ValIndex = sklearn.model_selection.train_test_split(range(0, len(lUniqueDirNoI)), test_size = fTestSize , shuffle = bShuffle, random_state = iSeed)
    np1TrainDir = lUniqueDirNoI[TrainIndex]
    np1ValDir = lUniqueDirNoI[ValIndex]
    # error check to ensure all patients have been accounted for
    if (len(np1TrainDir) + len(np1ValDir) + len(np1TestDir)) != len(npFiles) :
        raise ValueError('Num train, val, and test patients not equal to number of unique patients. Patient data was lost in the data generation')
    return np1TrainDir, np1ValDir, np1TestDir 
                                  
def CreateNFolds(npGroupList, iNFolds, bShuffle = True, seed = 1):
    """ Creates an 2 d numpy array of indexes with N arrays of arrays.
    The values of the list are NOT returned. This uses random.random package
    as np.random package is not thread safe
    
    Arguments:
        npGroupList: Input list that will be split into N arrays.
        iNFolds: Number of splits
        shuffle: Indicates if the groups will be shuffled or not
        seed: if shuffle = True this indicates what random seed, if any, will be used to
            initiate the shuffle.
            
    Returns:
        A 2D numpy array containing the indexes. For example if there are 10 groups 
        and iNFolds = 2, then the returned array would be [[1,2,3,4,5], [6,7,8,9,10]]
        if shuffling was False or [[10,5,9,3,6], [4,8,2,7,1]] if shuffling was true
    """
    # TODO: make error metrics for this
    ilIndexes = np.arange(len(npGroupList))
    if seed is not None:
        random.seed(seed)
    if bShuffle:
        random.shuffle(ilIndexes)
    return np.array_split(ilIndexes, iNFolds)

def FoldNSplit(npFiles, bShufflePatientOrder, iPatientOrderSeed, fTestSize, bShuffleSplitOrder,  iNumFoldsTotal, iFold):
    """
    This function splits npFiles into train,val and test sets.
    
    breaks npFiles into iNumFoldsTotal. This function will then select the
    iFold group as the testing data. It will then split the remaining groups into 
    training and validation groups. It will split fTestSize/1 percent of the remaining groups into the 
    validation set. The rest will be used for training.
    
    For example if there were 10 patients in npFiles. If iNumFoldsTotal=5, then 
    this function will split npFiles into 5 groups of two. If iFold=1 then the 1th group
    will be set aside for training. The 0,2,3,4 th groups will then be split. If fTestSize =0.3
    then 30% of the patients from groups 0,2,3,4 will be selected for validation. The rest will be used for
    training.
    
    The boolean arguments are used to determine if the patients will be shuffled.    
    
    
    """
    np2FoldIndexes = CreateNFolds(npFiles, iNumFoldsTotal, bShuffle = bShufflePatientOrder, seed = iPatientOrderSeed)
    # generate list of training, validatio and testing directories
    np1TrainDir, np1ValDir, np1TestDir = SplitDirs(npFiles, np2FoldIndexes, fTestSize, iFold, bShuffleSplitOrder)
    return np1TrainDir, np1ValDir, np1TestDir

def FlattenAndRemoveIndex(ListOfLists, NIndex):
    """ Flattens a list of lists after removing the Nth Index. Can only remove one index
    """
    ListOfLists =  ListOfLists[:NIndex] +  ListOfLists[NIndex+1:]
    return [item for sublist in ListOfLists for item in sublist]

def FlattenList(List):
    FlattenList = [item for sublist in List for item in sublist]
    return FlattenList


def CreateConfigDict(oConfig, sSection):
    """
    creates a dictionary from oConfig file, section.
    
    Converts any 'True' 'False' strings to booleans, 
    converts strings to ints if it's an '1' int in a string
    
    TODO: NEEDS TO BE ADJUSTED FOR OTHER INPUT TYPES
    """
    try:
        SectionDict = dict(oConfig.items(sSection))
    except KeyError as e:
        print("Augmentations catagory missing from ini file")
        print(e)

    for sItem in SectionDict:
        if SectionDict[sItem] == 'True':
            SectionDict[sItem] = True
        elif SectionDict[sItem] == 'False':
            SectionDict[sItem] = False
        # bunch of conditionals to convert the strings into floats if needed
        elif SectionDict[sItem] == 'height_shift_range' or 'width_shift_range': 
            SectionDict[sItem] = float(SectionDict[sItem])
        
    return SectionDict
        