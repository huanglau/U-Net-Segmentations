# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:27:10 2019
unit tests
Tests the error metrics functions.

Tests the GenErrorRates(npPred, npTruth)

@author: lhuang
"""

import unittest
import numpy as np
import src.ErrorMetrics as Error
import matplotlib.pyplot as plt

class TestGenErrorRates(unittest.TestCase):

#    def setUp(self):

        
    def test_AllTN(self):
        npTruthValues = np.array([0,0,0])
        npResults = np.array([0,0,0])
        [F1, recall, precision, ErrorRate] = Error.GenErrorRates(npResults, npTruthValues)
        npExpected = [ np.nan, np.nan, np.nan,0.0]
        self.assertTrue(np.isnan(F1))
        self.assertTrue(np.isnan(recall))
        self.assertTrue(np.isnan(precision))
        self.assertEqual(ErrorRate, npExpected[3])

    def test_AllFP(self):
        npTruthValues = np.array([0,0,0,0,0])
        npResults = np.array([1,1,1,1,1])
        [F1, recall, precision, ErrorRate] = Error.GenErrorRates(npResults, npTruthValues)
        npExpected = [ np.nan, np.nan, 0.0, 1.0]
        self.assertTrue(np.isnan(F1))
        self.assertTrue(np.isnan(recall))
        self.assertEqual(ErrorRate, npExpected[3]) 
        self.assertEqual(precision, npExpected[2]) 
    
    def test_AllFN(self):
        npTruthValues = np.array([1,1,1,1])
        npResults = np.array([0,0,0,0])
        [F1, recall, precision, ErrorRate] = Error.GenErrorRates(npResults, npTruthValues)
        npExpected = [ np.nan, 0.0, np.nan, 1.0]
        self.assertTrue(np.isnan(F1))
        self.assertTrue(np.isnan(precision))
        self.assertEqual(ErrorRate, npExpected[3]) 
        self.assertEqual(recall, npExpected[1])    
    
    def test_AllTP(self):
        npTruthValues = np.array([1,1,1,1])
        npResults = np.array([1,1,1,1])
        [F1, recall, precision, ErrorRate] = Error.GenErrorRates(npResults, npTruthValues)
        npExpected = [ 1.0,1.0, 1.0, 0.0]
        self.assertEqual([F1, recall, precision, ErrorRate], npExpected)    
    
    
    def test_Mixed1(self):
        npTruthValues = np.array([1,0,0,1])
        npResults = np.array([1,1,0,0])
        [F1, recall, precision, ErrorRate] = Error.GenErrorRates(npResults, npTruthValues)
        npExpected = [ 0.5, 0.5, 0.5, 0.5]
        self.assertEqual([F1, recall, precision, ErrorRate], npExpected)  

    def test_Mixed2(self):
        npTruthValues = np.array([1,0,0,1, 1, 1])
        npResults = np.array([1,1,0,0, 0, 0])
        [F1, recall, precision, ErrorRate] = Error.GenErrorRates(npResults, npTruthValues)
        npExpected = [ 2*0.25*0.5/(0.25+0.5), 1/(1+3), 1/(1+1), 4/6]
        self.assertEqual([F1, recall, precision, ErrorRate], npExpected)  
    
    def test_NonBoolean(self):
        npTruthValues = np.array([1,1,1,1])
        npResults = np.array([1,-1,1.2,.2])
        self.assertRaises(ValueError, Error.GenErrorRates, npResults, npTruthValues)

    def test_UnequalShapes(self):
        npTruthValues = np.array([1,1,1,1,1,1])
        npResults = np.array([1,-1,1.2,.2])
        self.assertRaises(ValueError, Error.GenErrorRates, npResults, npTruthValues)

if __name__ == "__main__":
    unittest.main()
