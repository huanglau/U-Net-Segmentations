# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:27:10 2019
unit tests
Tests the error metrics functions.

Tests the GenAUC(npPred, npTruth) function in ErrorMetrics.py

@author: lhuang
"""

import unittest
import numpy as np
import src.ErrorMetrics as Error
import matplotlib.pyplot as plt

class TestGenAUC(unittest.TestCase):

    def test_AllTN(self):
        """ case where everthing zero might make things weird
        """
        npTruthValues = np.array([0,0,0])
        npResults = np.array([0.2,0.3,0.4])
        [auc, fpr, tpr, thresholds] = Error.GenAUC(npResults, npTruthValues)
        self.assertTrue(np.isnan(auc))
#        self.assertTrue([0.   , 0.33333333, 1. ], fpr.tolist())
#        self.assertTrue([np.nan , np.nan, np.nan], fpr.tolist())    
#        self.assertEqual([1.4, 0.4, 0.2], thresholds.tolist())

    def test_AllCorrect(self):
        """ case where auc is one
        """
        npTruthValues = np.array([0,0,0,1,1,1])
        npResults = np.array([0.1,0.1,0.2, 0.9, 0.8, 0.7])
        [auc, fpr, tpr, thresholds] = Error.GenAUC(npResults, npTruthValues)
        self.assertTrue(auc, 1)
#        self.assertTrue([0. , 0. , 0. , 0.33333333, 1.], fpr.tolist())
#        self.assertTrue( [0. , 0.33333333 , 0. , 0.0, 1.], tpr.tolist())    
#        self.assertEqual([1.9, 0.9, 0.7, 0.2, 0.1], thresholds.tolist())

    def test_AllTP(self):
        """ case where auc is one
        """
        npTruthValues = np.array([1,1,1])
        npResults = np.array([0.9, 0.8, 0.7])
        [auc, fpr, tpr, thresholds] = Error.GenAUC(npResults, npTruthValues)
        self.assertTrue(np.isnan(auc))
#        self.assertTrue([0. , 0. , 0. , 0.33333333, 1.], fpr.tolist())
#        self.assertTrue( [0. , 0.33333333 , 0. , 0.0, 1.], tpr.tolist())    
#        self.assertEqual([1.9, 0.9, 0.7, 0.2, 0.1], thresholds.tolist())

    def test_AllFP(self):
        """ case where auc is one
        """
        npTruthValues = np.array([0,0,0])
        npResults = np.array([0.9, 0.8, 0.7])
        [auc, fpr, tpr, thresholds] = Error.GenAUC(npResults, npTruthValues)
        self.assertTrue(np.isnan(auc))
#        self.assertTrue([0. , 0. , 0. , 0.33333333, 1.], fpr.tolist())
#        self.assertTrue( [0. , 0.33333333 , 0. , 0.0, 1.], tpr.tolist())    
#        self.assertEqual([1.9, 0.9, 0.7, 0.2, 0.1], thresholds.tolist())

    def test_AllFN(self):
        """ case where auc is one
        """
        npTruthValues = np.array([1,1,1])
        npResults = np.array([0.1,0.1,0.2])
        [auc, fpr, tpr, thresholds] = Error.GenAUC(npResults, npTruthValues)
        self.assertTrue(np.isnan(auc))
#        self.assertTrue([0. , 0. , 0. , 0.33333333, 1.], fpr.tolist())
#        self.assertTrue( [0. , 0.33333333 , 0. , 0.0, 1.], tpr.tolist())    
#        self.assertEqual([1.9, 0.9, 0.7, 0.2, 0.1], thresholds.tolist())

    def test_OneTP(self):
        """ case where auc is one
        """
        npTruthValues = np.array([1,1,1,1,1,0])
        npResults = np.array([0.1,0.9,0.2,0.2,0.2, 0.2])
        [auc, fpr, tpr, thresholds] = Error.GenAUC(npResults, npTruthValues)
        self.assertTrue(auc, 0.5)
#        self.assertTrue([0. , 0. , 0. , 0.33333333, 1.], fpr.tolist())
#        self.assertTrue( [0. , 0.33333333 , 0. , 0.0, 1.], tpr.tolist())    
#        self.assertEqual([1.9, 0.9, 0.7, 0.2, 0.1], thresholds.tolist())
if __name__ == "__main__":
    unittest.main()
