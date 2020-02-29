# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:27:10 2019
unit tests
Tests the error metrics functions.

Tests the colour prediction map generator. That map generatesa map with 4 colours
where the colours correspond to TN, TP, FN, and TP


@author: lhuang
"""

import unittest
import src.ErrorMetrics as Error
import numpy as np

class TestColouredPredMap(unittest.TestCase):

    def setUp(self):
        self.npTP = [1,1,1] # white
        self.npTN = [0,0,0] # black
        self.npFP = [1,0,0] # red
        self.npFN = [0,0,1] # blue
        
    # 2D test cases
    def test_GenCase(self):
        npTrue = np.array([[[0],[1]],[[1],[0]]])
        npPred = np.array([[[0],[1]],[[0],[1]]])
        npColourMap = Error.ColouredPredMap(npPred, npTrue)
        npExpectedColourMap = np.array([[self.npTN, self.npTP],
                                     [self.npFN, self.npFP]], dtype= np.uint8)
        self.assertEqual(npColourMap[0].tolist(), npExpectedColourMap.tolist())

    def test_AllFalse(self):
        npTrue = np.array([[[0],[1]],[[1],[0]]])
        npPred = np.array([[[1],[0]],[[0],[1]]])
        npColourMap = Error.ColouredPredMap(npPred, npTrue)
        npExpectedColourMap = np.array([[self.npFP,self.npFN],
                                     [self.npFN,self.npFP]])
        self.assertEqual(npColourMap[0].tolist(), npExpectedColourMap.tolist())
        
    def test_AllFP(self):
        npTrue = np.array([[[0],[0]],[[0],[0]]])
        npPred = np.array([[[1],[1]],[[1],[1]]])
        npColourMap = Error.ColouredPredMap(npPred, npTrue)
        npExpectedColourMap = np.array([[self.npFP, self.npFP],
                                     [self.npFP, self.npFP]])
        self.assertEqual(npColourMap[0].tolist(), npExpectedColourMap.tolist()) 
        
    def test_AllFN(self):
        npTrue = np.array([[[1],[1]],[[1],[1]]])
        npPred = np.array([[[0],[0]],[[0],[0]]])
        npColourMap = Error.ColouredPredMap(npPred, npTrue)
        npExpectedColourMap = np.array([[self.npFN, self.npFN],
                                     [self.npFN, self.npFN]])
        self.assertEqual(npColourMap[0].tolist(), npExpectedColourMap.tolist()) 
        
    def test_AllTN(self):
        npTrue = np.array([[[0],[0]],[[0],[0]]])
        npPred = np.array([[[0],[0]],[[0],[0]]])
        npColourMap = Error.ColouredPredMap(npPred, npTrue)
        npExpectedColourMap = np.array([[self.npTN, self.npTN],
                                     [self.npTN, self.npTN]])
        self.assertEqual(npColourMap[0].tolist(), npExpectedColourMap.tolist()) 
        
    def test_AllTP(self):
        npTrue = np.array([[[1],[1]],[[1],[1]]])
        npPred = np.array([[[1],[1]],[[1],[1]]])
        npColourMap = Error.ColouredPredMap(npPred, npTrue)
        npExpectedColourMap = np.array([[self.npTP, self.npTP],
                                     [self.npTP, self.npTP]])
        self.assertEqual(npColourMap[0].tolist(), npExpectedColourMap.tolist()) 
        
    def test_3Ch(self):
        npTrue = np.array([[[0,0,1],[1,0,0]],[[0,0,1],[0,1,0]]])
        npPred = np.array([[[1],[1]],[[1],[1]]])
        npColourMap = Error.ColouredPredMap(npPred, npTrue[:,:,1:2])
        npExpectedColourMap = np.array([[self.npFP, self.npFP],
                                     [self.npFP, self.npTP]])
        self.assertEqual(npColourMap[0].tolist(), npExpectedColourMap.tolist()) 
    
if __name__ == "__main__":
    unittest.main()
