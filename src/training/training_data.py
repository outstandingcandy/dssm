import sys
import operator
from theano import tensor as T
import random
import cPickle
import numpy as np
from util.util import *
import scipy.sparse as sp
import cProfile
import theano

GOOD_TITLE_NUM = 1
BAD_TITLE_NUM = 4

def transfer_feature_dict2list(feature_dict):
    return sorted(feature_dict.iteritems(), key=operator.itemgetter(0))

def transfer_feature_sparse2dense(feature_dict):
    return feature_dict.todense()

def transfer_feature_dense2sparse(feature_matrix, feature_num):
    data = []
    indices = []
    indptr = [0]
    for feature_list in feature_matrix:
        for id, value in feature_list:
            data.append(value)
            indices.append(id)
        indptr.append(len(feature_list) + indptr[-1])
    return sp.csr_matrix((data, indices, indptr), shape=(len(feature_matrix), feature_num), dtype="float32")

def get_feature_from_string(feature_string):
    feature = []
    tokens = feature_string.strip().split(" ")
    for token in tokens:
        feature_pair = token.split(":")
        feature.append((int(feature_pair[0]), int(feature_pair[1])))
    return feature

def get_feature_from_string_v2(feature_string):
    feature = {}
    tokens = feature_string.strip().split(" ")
    for token in tokens:
        feature_pair = token.split(":")
        # feature[int(feature_pair[0])] = int(feature_pair[1])
        feature[int(feature_pair[0])] = 1
    return feature

class TrainingData(object):

    def __init__(self, combined_feature_file_name="../data/combined_feature", \
            combined_id_file_name = "../data/combined_id", sample_size = 64*1024):
        self.combined_feature_file = open(combined_feature_file_name)
        # self.combined_id_file = open(combined_id_file_name)
        self.feature_num = 62325
        self.sample_size = sample_size

    def clear(self):
        self.combined_feature_file.seek(0)
        # self.combined_id_file.seek(0)

    def load_sparse_training_data_v4(self, q, d):
        query_feature_list = []
        doc_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        for line in self.combined_feature_file:
            current_size += 1 
            if current_size > self.sample_size:
                break
            tokens = line.strip().split("\t")
            query_feature = get_feature_from_string_v2(tokens[0])
            query_feature_list.append(\
                    transfer_feature_dict2list(query_feature))
            positive_sample_feature = get_feature_from_string_v2(tokens[1])
            doc_feature_list.append(\
                    transfer_feature_dict2list(positive_sample_feature))
            for token in tokens[2:]:
                negative_sample_feature = get_feature_from_string_v2(token)
                doc_feature_list.append(\
                        transfer_feature_dict2list(negative_sample_feature))
        print_log("Transforming to sparse")
        # print_log("q\n%s" % query_feature_list[0])
        # print_log("d\n%s" % doc_feature_list[0])
        # print_log("d\n%s" % doc_feature_list[4])
        # print_log("q\n%s" % query_feature_list[11])
        # print_log("d\n%s" % doc_feature_list[55])
        q.set_value(transfer_feature_dense2sparse(\
                query_feature_list, self.feature_num), borrow=True)
        d.set_value(transfer_feature_dense2sparse(\
                doc_feature_list, self.feature_num), borrow=True)

    def load_sparse_training_data_v5(self):
        query_feature_list = []
        doc_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        for line in self.combined_feature_file:
            current_size += 1 
            if current_size > self.sample_size:
                break
            tokens = line.strip().split("\t")
            query_feature = get_feature_from_string(tokens[0])
            query_feature_list.append(query_feature)
            positive_sample_feature = get_feature_from_string(tokens[1])
            doc_feature_list.append(positive_sample_feature)
            for token in tokens[2:]:
                doc_feature_list.append(get_feature_from_string(token))
        print_log("Transforming to sparse")
        return transfer_feature_dense2sparse(query_feature_list, self.feature_num),\
                transfer_feature_dense2sparse(doc_feature_list, self.feature_num)

    def load_sparse_training_data_v6(self, q, d):
        query_feature_list = []
        doc_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        for line in self.combined_feature_file:
            current_size += 1 
            if current_size > self.sample_size:
                break
            tokens = line.strip().split("\t")
            query_feature = get_feature_from_string_v2(tokens[0])
            query_feature_list.append(\
                    transfer_feature_dict2list(query_feature))
            positive_sample_feature = get_feature_from_string_v2(tokens[1])
            doc_feature_list.append(\
                    transfer_feature_dict2list(positive_sample_feature))
            for token in tokens[2:]:
                negative_sample_feature = get_feature_from_string_v2(token)
                doc_feature_list.append(\
                        transfer_feature_dict2list(negative_sample_feature))
        print_log("Transforming to sparse")
        print_log("q\n%s" % query_feature_list[0])
        print_log("d\n%s" % doc_feature_list[0])
        print_log("d\n%s" % doc_feature_list[4])
        print_log("q\n%s" % query_feature_list[11])
        print_log("d\n%s" % doc_feature_list[55])
        q.set_value(transfer_feature_sparse2dense(\
                transfer_feature_dense2sparse(\
                query_feature_list, self.feature_num)), borrow=True)
        d.set_value(transfer_feature_sparse2dense(\
                transfer_feature_dense2sparse(\
                doc_feature_list, self.feature_num)), borrow=True)

if __name__ == "__main__":
    # Prepare for theano
    training_data = TrainingData()
    cProfile.run("training_data.load_sparse_training_data_v3()")
