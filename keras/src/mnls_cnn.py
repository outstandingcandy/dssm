from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Dropout, Lambda, Flatten, Activation, Reshape, Input, Embedding, merge, Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D
from keras.optimizers import SGD, RMSprop
from keras import backend as K

import training_data
from multiple_fully_connected_layer import *

import theano
from theano import tensor as T
from theano.ifelse import ifelse
from theano.compile.nanguardmode import NanGuardMode
from theano.compile.debugmode import DebugMode
theano.config.optimizer='None'
theano.config.exception_verbosity='high'

from my_embedding_layer import MyEmbedding

def dssm_loss(d, y):
    q = y[:, :y.shape[1]/3]
    d_positive = y[:, y.shape[1]/3:2*y.shape[1]/3]
    d_negative = y[:, 2*y.shape[1]/3:]
    eps = 0.00001
    inf = 100000
    q_len = K.sqrt(K.sum(K.square(q), axis=1))
    d_positive_len = K.sqrt(K.sum(K.square(d_positive), axis=1))
    d_negative_len = K.sqrt(K.sum(K.square(d_negative), axis=1))
    positive_cosine_distance = K.sum(q * d_positive, axis=1) / (q_len * d_positive_len)
    negative_cosine_distance = K.sum(q * d_negative, axis=1) / (q_len * d_negative_len)
    all_cosine_distance = K.exp(positive_cosine_distance) + K.exp(negative_cosine_distance)
    positive_cosine_distance = K.exp(positive_cosine_distance) / (all_cosine_distance)
    loss = -K.mean(K.log(positive_cosine_distance))
    return ifelse(T.isnan(loss), 0., loss)
    # return -K.max(positive_cosine_distance)

def generate_weight(weight_file_name, word2vector_size=200):
    weight_matrix = np.ones((term_num+1, word2vector_size)) * 0.0001
    for line in open(weight_file_name):
        tokens = line.strip().split(' ')
        if len(tokens) != word2vector_size + 1:
            print("ERROR: word2vector")
        word_id = int(tokens[0]) + 1
        for i in range(word2vector_size):
            weight_matrix[word_id][i] = float(tokens[i+1])
    print("weight size %s" % str(weight_matrix.shape))
    return weight_matrix

def create_word_embedding_network(input_dim):
    # weights = generate_weight("/home/search/dssm/data/vector.overlap.nolng")
    weights = np.random.randn(term_num+1, 200)
    seq = Sequential()
    seq.add(MyEmbedding(weights.shape[0], weights.shape[1], weights=[weights], mask_zero=False, input_length=input_dim, trainable=False))
    seq.add(Reshape((1, input_dim, word2vector_size)))
    seq.add(Convolution2D(3, 3, 1, border_mode='same'))
    seq.add(MaxPooling2D(pool_size=(2, 1)))
    seq.add(Flatten())
    seq.add(Dense(300, activation = "relu"))
    seq.add(Dense(128, activation = "relu"))
    return seq

def create_mpl_network(input_dim):
    seq = Sequential()
    seq.add(Dense(300, input_dim = input_dim, activation = "relu"))
    seq.add(Dense(128, activation = "relu"))
    seq.add(Dense(1))
    seq.add(Activation('sigmoid'))
    return seq

# training_data_file_name = "/home/search/dssm/data/term_feature_201602x"
training_data_file_name = "/home/search/dssm/data/term_feature_v2_20160xxx"
sample_size = 1024*4
negative_d_num = 1
input_dim = 20
term_num = 1236275
word2vector_size = 200

if __name__ == "__main__":
    # network definition
    q = Input(shape=(input_dim, ), name="input_q", dtype="int32")
    d_positive = Input(shape=(input_dim, ), name="input_d_positive", dtype="int32")
    d_negative = Input(shape=(input_dim, ), name="input_d_negative", dtype="int32")
    # word_embedding network
    word_embedding = create_word_embedding_network(input_dim)
    embedding_q = word_embedding(q)
    embedding_d_positive = word_embedding(d_positive)
    embedding_d_negative = word_embedding(d_negative)
    merged = merge([embedding_q, embedding_d_positive, embedding_d_negative], mode='concat', name="triple_sentence_vector")
    final_model = Model(input=[q, d_positive, d_negative], output=[merged])

    # train
    sgd = SGD()
    print("compile network")
    final_model.compile(loss={'triple_sentence_vector': dssm_loss}, optimizer=sgd)
    # final_model.compile(loss={'triple_sentence_vector': dssm_loss_v2}, optimizer=sgd, mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False))
    # final_model.compile(loss={'triple_sentence_vector': dssm_loss_v2}, optimizer=sgd, mode=theano.compile.MonitorMode(post_func=detect_nan))
    # final_model.compile(loss={'triple_sentence_vector': dssm_loss_v2}, optimizer=sgd, mode=DebugMode(check_py_code=False))
    # final_model.compile(loss={'triple_sentence_vector': dssm_loss_v2}, optimizer=sgd, mode=DebugMode(check_py_code=False))

    for epoch in range(20):
        training_data_set =\
                training_data.TrainingData(\
                        combined_feature_file_name=training_data_file_name,\
                        sample_size=sample_size)
        final_model.fit_generator(training_data_set.generate_training_data_v2(negative_d_num), samples_per_epoch=1024*1024*16, nb_epoch=1, verbose=1)
        json_string = final_model.to_json()
        open('wec_model_architecture_v2.%d.json' % (epoch), 'w').write(json_string)
        final_model.save_weights('wec_model_weights_v2.%d.h5' % (epoch), overwrite=True)

