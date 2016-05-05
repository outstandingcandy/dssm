# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import sys

import scipy.sparse as sp

import theano
from theano import tensor as T

from keras import activations, initializations, regularizers, constraints
from keras import backend as K
from keras.engine import InputSpec, Layer, Merge
from keras.regularizers import ActivityRegularizer
from keras.layers.embeddings import Embedding

class MyEmbedding(Embedding):
    def call(self, x, mask=None):
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
            W[0].fill(0.0001)
        out = K.gather(W, x)
        return out
