import sys
import random
import cPickle
import theano
import numpy as np
from util import *
import scipy.sparse as sp
import cProfile

GOOD_TITLE_NUM = 1
BAD_TITLE_NUM = 4

class TrainingData(object):

    def __init__(self, combined_feature_file_name="../data/combined_feature", \
            combined_id_file_name = "../data/combined_id"):
        self.combined_feature_file = open(combined_feature_file_name)
        self.combined_id_file = open(combined_id_file_name)
        self.feature_num = 47947

    def clear(self):
        self.combined_feature_file.seek(0)
        self.combined_id_file.seek(0)

    def transfer_feature_dict2list(self, feature_dict):
        feature_list = [0] * self.feature_num
        for feature_id, feature_value in feature_dict.items():
            feature_list[feature_id] = feature_value
        return feature_list

    def transfer_feature_dense2sparse(self, feature_matrix):
        data = []
        indices = []
        indptr = [0]
        for feature_list in feature_matrix:
            for id, value in feature_list:
                data.append(value)
                indices.append(id)
            indptr.append(len(feature_list) + indptr[-1])
        return sp.csr_matrix((data, indices, indptr), shape=(len(feature_matrix), self.feature_num), dtype=theano.config.floatX)

    def get_feature_from_string(self, feature_string):
        feature = []
        tokens = feature_string.strip().split(" ")
        for token in tokens:
            feature_pair = token.split(":")
            feature.append((int(feature_pair[0]), int(feature_pair[1])))
        return feature

    def load_sparse_training_data(self, sample_size):
        query_feature_list = []
        positive_sample_feature_list = []
        negative_sample_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        for line in self.combined_feature_file:
            current_size += 1 
            if current_size > sample_size:
                break
            tokens = line.strip().split("\t")
            query_feature_list.append( \
                    self.get_feature_from_string(tokens[0]))
            positive_sample_feature_list.append( \
                    self.get_feature_from_string(tokens[1]))
            for token in tokens[2:]:
                negative_sample_feature_list.append( \
                        self.get_feature_from_string(token))
        print_log("Transforming to sparse")
        return theano.shared(self.transfer_feature_dense2sparse(query_feature_list), borrow=True), \
            theano.shared(self.transfer_feature_dense2sparse(positive_sample_feature_list), borrow=True), \
            theano.shared(self.transfer_feature_dense2sparse(negative_sample_feature_list), borrow=True)

    def load_sparse_training_data_v2(self, sample_size):
        query_feature_list = []
        positive_sample_feature_list = []
        negative_sample_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        for line in self.combined_feature_file:
            current_size += 1 
            if current_size > sample_size:
                break
            tokens = line.strip().split("\t")
            query_feature = self.get_feature_from_string(tokens[0])
            positive_sample_feature = self.get_feature_from_string(tokens[1])
            for token in tokens[2:]:
                query_feature_list.append(query_feature)
                positive_sample_feature_list.append(positive_sample_feature)
                negative_sample_feature_list.append( \
                        self.get_feature_from_string(token))
        print_log("Transforming to sparse")
        return theano.shared(self.transfer_feature_dense2sparse(query_feature_list), borrow=True), \
            theano.shared(self.transfer_feature_dense2sparse(positive_sample_feature_list), borrow=True), \
            theano.shared(self.transfer_feature_dense2sparse(negative_sample_feature_list), borrow=True)

    def load_sparse_training_data_v3(self, sample_size):
        query_feature_list = []
        doc_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        for line in self.combined_feature_file:
            current_size += 1 
            if current_size > sample_size:
                break
            tokens = line.strip().split("\t")
            query_feature = self.get_feature_from_string(tokens[0])
            query_feature_list.append(query_feature)
            positive_sample_feature = self.get_feature_from_string(tokens[1])
            doc_feature_list.append(positive_sample_feature)
            for token in tokens[2:]:
                doc_feature_list.append(self.get_feature_from_string(token))
        print_log("Transforming to sparse")
        return theano.shared(self.transfer_feature_dense2sparse(query_feature_list), borrow=True), \
            theano.shared(self.transfer_feature_dense2sparse(doc_feature_list), borrow=True)

if __name__ == "__main__":
    # Prepare for theano
    training_data = TrainingData()
    cProfile.run("training_data.load_sparse_training_data_v3(4096*16)")
