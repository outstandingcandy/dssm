# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import sys

import scipy.sparse as sp

import theano
from theano import sparse as TS
from theano import tensor as T

from keras import activations, initializations, regularizers, constraints
from keras import backend as K
from keras.engine import InputSpec, Layer, Merge
from keras.regularizers import ActivityRegularizer
from keras.layers.core import Dense

class MultipleFullyConnectedLayer(Dense):
    def call(self, x, mask=None):
        output_list = []
        # for i in range(x.shape[1] / self.W.shape[0]):
        for i in range(2):
            output_list.append(self.activation(K.dot(x[i*self.W.shape[1] : (i+1)*self.W.shape[1]], self.W) + self.b))
        return K.concatenate(output_list, axis=1)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim * 2)
