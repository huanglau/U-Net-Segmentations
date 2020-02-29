# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:27:10 2019
unit tests
Tests the error metrics functions.

Tests error metrics, confusion matrix generator. 
Generates confusion matrixes from two inpuit images. One truth, and one pred

@author: lhuang
"""

import unittest
import src.ErrorMetrics as Error
import numpy as np

class ConfusionTests(unittest.TestCase):

    def setUp(self):
        self.truth = np.zeros(1)
        
    def test_allFalseNeg(self):
        npTestImgTruth1 = np.ones((3,3))
        npTestImgGen1 = np.zeros((3,3))
        npError = Error.ConfMatrix(npTestImgTruth1, npTestImgGen1)
        npExpectError = [0,0,9,0]
        self.assertEqual(npExpectError[0], npError[0])
        self.assertEqual(npExpectError[1:], npError[1:])
        
    def test_allFalsePos(self):
        npTestImgTruth1 = np.zeros((3,3))
        npTestImgGen1 = np.ones((3,3))
        npError = Error.ConfMatrix(npTestImgTruth1, npTestImgGen1)
        npExpectError = [0,9,0,0]
        self.assertEqual(npExpectError[0], npError[0])
        self.assertEqual(npExpectError[1:], npError[1:])

    def test_allTrueNeg(self):
        npTestImgTruth1 = np.zeros((3,3))
        npTestImgGen1 = np.zeros((3,3))
        npError = Error.ConfMatrix(npTestImgTruth1, npTestImgGen1)
        npExpectError = [9,0,0,0]
        self.assertEqual(npExpectError[0], npError[0])
        self.assertEqual(npExpectError[1:], npError[1:])
       
    def test_allTruePos(self):
        npTestImgTruth1 = np.ones((3,3))
        npTestImgGen1 = np.ones((3,3))
        npError = Error.ConfMatrix(npTestImgTruth1, npTestImgGen1)
        npExpectError = [0,0,0,9]
        self.assertEqual(npExpectError[0], npError[0])
        self.assertEqual(npExpectError[1:], npError[1:])

    def test_mixed(self):
        npTestImgTruth1 = np.array([[0,0,1],[0,0,1],[1,0,1]])
        npTestImgGen1 = np.array([[0,0,1],[1,1,0],[0,0,0]])
        npError = Error.ConfMatrix(npTestImgTruth1, npTestImgGen1)
        #tn, fp, fn, tp
        npExpectError = [3,2,3,1]
        self.assertEqual(npExpectError[0], npError[0])
        self.assertEqual(npExpectError[1:], npError[1:])        
    
if __name__ == "__main__":
    unittest.main()
