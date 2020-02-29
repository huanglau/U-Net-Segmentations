# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:27:10 2019
unit tests
Tests the error metrics functions.

Tests the confusion map generator that summs up the number of pixels of any given 


npTNRange, npFPRange, npFNRange, npTPRange, npDonuts = Error.ConfMatrixFromErrorMap(npColourPredMapRange, lLegendRange)               



Uses the default kernel
kernel = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]], np.uint8)

@author: lhuang
"""

import unittest
import numpy as np
import src.ErrorMetrics as Error
import matplotlib.pyplot as plt

class TestColouredPredMapRangeDonutsUpSample(unittest.TestCase):

    def setUp(self):
        self.grey = [145,145,145]
        self.npTP = [255,255,255] # white
        self.npTN = [0, 1, 0]# blk
        self.npFP = [255, 0, 0]  # red
        self.npFN = [0, 0, 255]  # blue
        
    def test_TP(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey]
        npExpectedColourMap = np.array([[self.npTP, self.npTP, self.grey, self.grey, self.npTN, self.npTN],
                                     [self.npTP, self.npTP,  self.grey, self.grey, self.npTN, self.npTN],
                                     [self.grey, self.grey,  self.grey, self.grey, self.npTN, self.npTN],
                                     [self.grey, self.grey,  self.grey, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN,  self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN,  self.npTN, self.npTN, self.npTN, self.npTN]], dtype= np.uint8)
        npExpectedCounts = [21, 0, 0, 4, 11] # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts)
        
    def test_TPDiffColours(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey, [0,0,0]]
        npExpectedColourMap = np.array([[self.npTP, self.npTP, self.grey, self.grey, self.npTN, self.npTN],
                                     [self.npTP, self.npTP,  self.grey, self.grey, self.npTN, self.npTN],
                                     [self.grey, self.grey,  self.grey,  [0,0,0], self.npTN, self.npTN],
                                     [self.grey, self.grey,  self.grey, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN,  self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN,  self.npTN,   [0,0,0],   [0,0,0], self.npTN]], dtype= np.uint8)
        npExpectedCounts = [19, 0, 0, 4, 10, 3] # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts) 
     
    def test_FP(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey]
        npExpectedColourMap = np.array([[self.npTP, self.npTP, self.grey, self.grey, self.npTN, self.npTN],
                                     [self.npTP, self.npTP,  self.grey, self.grey, self.npTN, self.npTN],
                                     [self.grey, self.grey,  self.grey, self.grey, self.npTN, self.npTN],
                                     [self.grey, self.grey,  self.grey, self.npFP, self.npFP, self.npFP],
                                     [self.npTN, self.npTN,  self.npTN, self.npFP, self.npFP, self.npFP],
                                     [self.npTN, self.npTN,  self.npTN, self.npFP, self.npFP, self.npFP]], dtype= np.uint8)
        npExpectedCounts = [12, 9, 0, 4, 11]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts)

    def test_FPMoreColours(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey, [0,0,0], [0,0,1]]
        npExpectedColourMap = np.array([[[0,0,1], self.npTP, self.grey, self.grey, self.npTN, self.npTN],
                                     [self.npTP, self.npTP,  self.grey, self.grey, self.npTN, self.npTN],
                                     [self.grey, self.grey,  self.grey, self.grey, self.npTN, self.npTN],
                                     [[0,0,1], self.grey,  self.grey, self.npFP, self.npFP, self.npFP],
                                     [[0,0,1], self.npTN,  self.npTN, self.npFP, self.npFP, self.npFP],
                                     [[0,0,1], self.npTN,  self.npTN, self.npFP, self.npFP, self.npFP]], dtype= np.uint8)
        npExpectedCounts = [10, 9, 0, 3, 10, 0, 4]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts)

    def test_FN(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey]
        npExpectedColourMap = np.array([[self.npFP, self.npFP, self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npFP, self.npFP,  self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.grey, self.grey,  self.grey, self.npFP, self.npFP, self.npFP],
                                     [self.grey, self.grey,  self.grey, self.grey, self.npTN, self.npTN],
                                     [self.npFN, self.npFN,  self.grey, self.grey, self.npTN, self.npTN],
                                     [self.npFN, self.npFN,  self.grey, self.grey, self.npTN, self.npTN]], dtype= np.uint8)
        npExpectedCounts = [6, 15, 4, 0, 11]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts)

    def test_AllTN(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey]
        npExpectedColourMap = np.array([[self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN]], dtype= np.uint8)
        npExpectedCounts = [6*6, 0, 0, 0, 0]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts)
    
    def test_AllFP(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey]
        npExpectedColourMap = np.array([[self.npFP, self.npFP, self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npFP, self.npFP, self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npFP, self.npFP, self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npFP, self.npFP, self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npFP, self.npFP, self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npFP, self.npFP, self.npFP, self.npFP, self.npFP, self.npFP]], dtype= np.uint8)
        npExpectedCounts = [0, 6*6, 0, 0, 0]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts)
    
    def test_AllFN(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey]
        npExpectedColourMap = np.array([[self.npFN, self.npFN, self.npFN, self.npFN, self.npFN, self.npFN],
                                     [self.npFN, self.npFN, self.npFN, self.npFN, self.npFN, self.npFN],
                                     [self.npFN, self.npFN, self.npFN, self.npFN, self.npFN, self.npFN]], dtype= np.uint8)
        npExpectedCounts = [0, 0, 6*3, 0, 0]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts)
    
    def test_AllTP(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey]
        npExpectedColourMap = np.array([[self.npTP, self.npTP, self.npTP, self.npTP, self.npTP],
                                     [self.npTP, self.npTP, self.npTP, self.npTP, self.npTP],
                                     [self.npTP, self.npTP, self.npTP, self.npTP, self.npTP]], dtype= np.uint8)
        npExpectedCounts = [0, 0, 0, 5*3, 0]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts)

    def test_AllGrey(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey]
        npExpectedColourMap = np.array([[self.grey, self.grey],
                                     [self.grey, self.grey],
                                     [self.grey, self.grey]], dtype= np.uint8)
        npExpectedCounts = [0, 0, 0, 0, 6]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts)
    
    def test_NoGrey(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP]
        npExpectedColourMap = np.array([[self.npFP, self.npFP, self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npFP, self.npFP,  self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npTP, self.npTP,  self.npTP, self.npFP, self.npFP, self.npFP],
                                     [self.npTP, self.npTP,  self.npTP, self.npTP, self.npTN, self.npTN],
                                     [self.npFN, self.npFN,  self.npTP, self.npTP, self.npTN, self.npTN],
                                     [self.npFN, self.npFN,  self.npTP, self.npTP, self.npTN, self.npTN]], dtype= np.uint8)
        npExpectedCounts = [6, 15, 4, 11]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts) 
    
    def test_WrongLegend(self):
        """ situation where legend is not all possible pixel values
        """
        lLegend = [self.npTN, self.npFP, self.npFN]
        npExpectedColourMap = np.array([[self.npFP, self.npFP, self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npFP, self.npFP,  self.npFP, self.npFP, self.npFP, self.npFP],
                                     [self.npTP, self.npTP,  self.npTP, self.npFP, self.npFP, self.npFP],
                                     [self.npTP, self.npTP,  self.npTP, self.npTP, self.npTN, self.npTN],
                                     [self.npFN, self.npFN,  self.npTP, self.npTP, self.npTN, self.npTN],
                                     [self.npFN, self.npFN,  self.npTP, self.npTP, self.npTN, self.npTN]], dtype= np.uint8)
        self.assertRaises(ValueError, Error.ConfMatrixFromErrorMap, npExpectedColourMap, lLegend)
    
    def test_EmptyMap(self):
        """ situation where legend is not all possible pixel values
        """
        lLegend = [self.npTN, self.npFP, self.npFN]
        npExpectedColourMap = np.array([[]], dtype= np.uint8)
        self.assertRaises(ValueError, Error.ConfMatrixFromErrorMap, npExpectedColourMap, lLegend)
    
    def test_AllLarger(self):
        lLegend = [self.npTN, self.npFP, self.npFN, self.npTP, self.grey]
        npExpectedColourMap = np.array([[self.npFN, self.npFN, self.npFN, self.npFN, self.npFN, self.grey, self.grey, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npFN, self.npFN, self.npFN, self.npFN, self.npFN, self.grey, self.grey, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npFN, self.npFN, self.npFN, self.npFN, self.npFN, self.grey, self.grey, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTP, self.npTP, self.npTP, self.npTP, self.npTP, self.grey, self.grey, self.npFP, self.npFP, self.npTN, self.npTN, self.npTN],
                                     [self.npTP, self.npTP, self.npTP, self.npTP, self.npTP, self.grey, self.grey, self.npFP, self.npFP, self.npTN, self.npTN, self.npTN],
                                     [self.npTP, self.npTP, self.npTP, self.npTP, self.npTP, self.grey, self.grey, self.npFP, self.npFP, self.npTN, self.npTN, self.npTN],
                                     [self.npFN, self.npFN, self.npFN, self.npFN, self.npFN, self.grey, self.grey, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npFN, self.npFN, self.npFN, self.npFN, self.npFN, self.grey, self.grey, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.grey, self.grey, self.grey, self.grey, self.grey, self.grey, self.grey, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.grey, self.grey, self.grey, self.grey, self.grey, self.grey, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN],
                                     [self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN, self.npTN]], dtype= np.uint8)
        npExpectedCounts = [9*5+6*4, 6, 9+6*2+4, 15, 6*5-1]  # tn, fp, fn, tp, range
        self.assertEqual(Error.ConfMatrixFromErrorMap(npExpectedColourMap, lLegend), npExpectedCounts) 

if __name__ == "__main__":
    unittest.main()
