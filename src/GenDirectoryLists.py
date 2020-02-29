# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:17:04 2019

This function creates the text files to be used in the analysis
The text files contains a list of directory pairs seperated by 
group ID 

@author: lhuang
"""

import glob
import numpy as np
import os


def GenerateDirList(sDirClasses1, sDirClasses2, sDirFile):
    """
    Generates a text file that lists out the directories to be used by the training algorithm.
    Inputs:    
        sDirClasses1: string. Directory to files containg class a
        sDirClasses2: string. Directory to files containg class a
        sDirFile: directory and file name of the text file to be generated
    outputs:
        none. outputs a file at sDirFile
    Lists the two pairs of images directories (i.e. invivo and ex vivo)
    where each row is a different patient dir and the colours are the image types.
    Assumes that the naming system is the same in the two directories
    """
    np1FilesClasses1 = np.sort(np.array(glob.glob(sDirClasses1)))
    np1FilesClasses2 = np.sort(np.array(glob.glob(sDirClasses2)))
    lFileNames1 = [os.path.split(sDir)[-1] for sDir in np1FilesClasses1]
    lFileNames2 = [os.path.split(sDir)[-1] for sDir in np1FilesClasses1] 
    if lFileNames2 != lFileNames1 or len(lFileNames1) != len(lFileNames2):
        raise ValueError('Subdirectories are not properly paired')
    npFiles = np.stack((np1FilesClasses1, np1FilesClasses2))
    file = open(sDirFile, 'w+')    
    # iterate through image pair directories
    for i in range(0, np.shape(npFiles)[1]):
        print("{},{}\n".format(npFiles[0,i], npFiles[1,i]))
        file.write("{},{}\n".format(npFiles[0,i], npFiles[1,i]))
    file.close()
    

GenerateDirList('../ExampleSegmentations/sky/data/*',
                '../ExampleSegmentations/sky/groundtruth/*', 
                '../ConfigFiles/SegmentationDataset.txt')
    
