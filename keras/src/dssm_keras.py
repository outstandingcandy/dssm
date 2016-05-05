from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Dropout, Lambda, Flatten, Activation, Reshape, Input, SparseInput, Embedding, merge, Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D
from keras.optimizers import SGD, RMSprop
from keras import backend as K

import training_data
from sparse_fully_connected_layer import *
from theano import tensor as T
from theano.ifelse import ifelse

def print_features(features):
    feature_string = ""
    for i in range(len(features)):
        if features[i] != 0:
            feature_string += str(i) + ":" + str(features[i]) + " "
    print(feature_string.strip())

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

def create_mlp_network(input_dim):
    seq = Sequential()
    seq.add(SparseFullyConnectedLayer(300, input_dim = input_dim, activation = "relu"))
    # seq.add(Dense(300, input_dim = input_dim, activation = "relu"))
    seq.add(Dense(300, activation = "relu"))
    seq.add(Dense(128, activation = "relu"))
    return seq

training_data_file_name = "/home/search/dssm/data/term_feature_v2_20160xxx"
sample_size = 1024*4
negative_d_num = 1
input_dim = 1236275

if __name__ == "__main__":
    # network definition
    q = SparseInput(shape=(input_dim, ), name="input_q", dtype=K.floatx())
    d_positive = SparseInput(shape=(input_dim, ), name="input_d_positive", dtype=K.floatx())
    d_negative = SparseInput(shape=(input_dim, ), name="input_d_negative", dtype=K.floatx())
    # q = Input(shape=(input_dim, ), name="input_q", dtype=K.floatx())
    # d_positive = Input(shape=(input_dim, ), name="input_d_positive", dtype=K.floatx())
    # d_negative = Input(shape=(input_dim, ), name="input_d_negative", dtype=K.floatx())
    # word_embedding network
    mlp = create_mlp_network(input_dim)
    mlp_q = mlp(q)
    mlp_d_positive = mlp(d_positive)
    mlp_d_negative = mlp(d_negative)
    merged = merge([mlp_q, mlp_d_positive, mlp_d_negative], mode='concat', name="triple_sentence_vector")
    final_model = Model(input=[q, d_positive, d_negative], output=[merged])

    # train
    sgd = SGD()
    print("compile network")
    final_model.compile(loss={'triple_sentence_vector': dssm_loss}, optimizer=sgd, sparse_input=True)
    # final_model.compile(loss={'triple_sentence_vector': dssm_loss}, optimizer=sgd, sparse_input=False)
    # final_model.compile(loss={'triple_sentence_vector': dssm_loss_v2}, optimizer=sgd, mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False))
    # final_model.compile(loss={'triple_sentence_vector': dssm_loss_v2}, optimizer=sgd, mode=theano.compile.MonitorMode(post_func=detect_nan))
    # final_model.compile(loss={'triple_sentence_vector': dssm_loss_v2}, optimizer=sgd, mode=DebugMode(check_py_code=False))
    # final_model.compile(loss={'triple_sentence_vector': dssm_loss_v2}, optimizer=sgd, mode=DebugMode(check_py_code=False))

    for epoch in range(20):
        training_data_set =\
                training_data.TrainingData(\
                        combined_feature_file_name=training_data_file_name,\
                        sample_size=sample_size)
        final_model.fit_generator(training_data_set.generate_sparse_training_data(negative_d_num), samples_per_epoch=1024*1024*16, nb_epoch=1, verbose=1)
        # final_model.fit_generator(training_data_set.generate_dense_training_data(negative_d_num), samples_per_epoch=1024*1024*16, nb_epoch=1, verbose=1)
        json_string = final_model.to_json()
        open('wec_model_architecture_v2.%d.json' % (epoch), 'w').write(json_string)
        final_model.save_weights('wec_model_weights_v2.%d.h5' % (epoch), overwrite=True)

