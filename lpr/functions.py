"""
****Needed characters*******************
****Activation function definitions*****
*****common for all files***************
"""

import numpy
from scipy.special import expit

__all__ = (
    'DIGITS',
    'LETTERS',
    'CHARS',
    'sigmoid',
    'softmax',
)



DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS

def softmax(a):
    exps = expit(a)
    #exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
    exps = expit(-a)
    return 1. / (1. + exps)
   # return 1. / (1. + numpy.exp(-a))
