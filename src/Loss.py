"""
Created on Thu Feb 28 15:09:26 2019
loss and error metrics functionst o be used in keras
while training. Can be used as a validation error metric
or as loss functions
@author: lhuang
"""

import keras.backend as K

def euclidean_distance(y_true, y_pred):
    sum_square = K.sum(K.square(y_true-y_pred), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def Recall(y_true, y_pred):
    """ calculates error metric recall: TP/(TP+FN)
    
    Assumes a binary classification, and that trues are 1's'
    """
    
    # get number of true positives
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    # all possible positives
    AllP = K.sum(y_true)
    
    # return recall
    return  TP / (AllP + K.epsilon())

    
def Precision(y_true, y_pred):
    """PCalculates error metric precision: TP/(TP+FP)
    
    Assumes a binary classification, and that trues are 1's'

    """
    # get true positives
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    # get TP+FP aka the predicted positives. Rounds the predictions
    TPFP = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    # return precision
    return TP / (TPFP + K.epsilon())
    
def f1(y_true, y_pred):
    """https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    https://en.wikipedia.org/wiki/F1_score
    Analysis of a binary classifier. 
    """
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def dyn_weighted_bincrossentropy(true, pred):
    """
    Calculates weighted binary cross entropy. The weights are determined dynamically
    by the balance of each category. This weight is calculated for each batch.
    
    The weights are calculted by determining the number of 'pos' and 'neg' classes 
    in the true labels, then dividing by the number of total predictions.
    
    For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
    These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
    1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.
    
    This can be useful for unbalanced catagories.
    """
    # get the total number of inputs
    num_pred = K.sum(K.cast(pred < 0.5, true.dtype)) + K.sum(true)
    
    # get weight of values in 'pos' category
    zero_weight =  K.sum(true)/ num_pred +  K.epsilon() 
    
    # get weight of values in 'false' category
    one_weight = K.sum(K.cast(pred < 0.5, true.dtype)) / num_pred +  K.epsilon()

    # calculate the weight vector
    weights =  (1.0 - true) * zero_weight +  true * one_weight 
    
    # calculate the binary cross entropy
    bin_crossentropy = K.binary_crossentropy(true, pred)
    
    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return K.mean(weighted_bin_crossentropy)


def weighted_bincrossentropy(true, pred, weight_zero = 0.25, weight_one = 1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.
        
    This can be useful for unbalanced catagories.
    
    Adjust the weights here depending on what is required.
    
    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
        will be penalize 10 times as much as false negatives.
    """
  
    # calculate the binary cross entropy
    bin_crossentropy = K.binary_crossentropy(true, pred)
    
    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return K.mean(weighted_bin_crossentropy)
