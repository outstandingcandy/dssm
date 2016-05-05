import sys
import operator
from theano import tensor as T
import random
import cPickle
import numpy as np
from util import *
import scipy.sparse as sp
import cProfile
import theano

GOOD_TITLE_NUM = 1
BAD_TITLE_NUM = 4

def transfer_feature_dict2list(feature_dict):
    return sorted(feature_dict.iteritems(), key=operator.itemgetter(0))

def transfer_feature_sparse2dense(feature_matrix, feature_num):
    dense_feature_list = np.empty((len(feature_matrix), feature_num))
    for i in range(len(feature_matrix)):
        feature_list = get_feature_from_string_v2(feature_matrix[i])
        for id, value in feature_list.items():
            dense_feature_list[i][id] = value
    return dense_feature_list

def get_feature_index(feature_matrix, feature_num, feature_length = 20):
    feature_index_list = []
    for feature in feature_matrix:
        feature_index_list.append(get_feature_index_from_string(feature))
    return np.asarray(feature_index_list)


def transfer_feature_dense2sparse(feature_matrix, feature_num):
    data = []
    indices = []
    indptr = [0]
    for feature_string in feature_matrix:
        feature_list = get_feature_from_string_v2(feature_string)
        for id, value in feature_list.items():
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

def get_feature_index_from_string(feature_string, feature_length=20):
    feature_index = [0] * feature_length
    tokens = feature_string.strip().split(" ")
    for i in range(len(tokens[:feature_length])):
        feature_pair = tokens[i].split(":")
        feature_index[i] = int(feature_pair[0]) + 1
        if np.isinf(feature_index[i]):
            print feature_string
    return feature_index

class TrainingData(object):

    def __init__(self, combined_feature_file_name="../data/combined_feature", \
            combined_id_file_name = "../data/combined_id", sample_size = 64*1024, feature_num = 1236275):
        self.combined_feature_file = open(combined_feature_file_name)
        # self.combined_id_file = open(combined_id_file_name)
        self.feature_num = feature_num
        self.sample_size = sample_size

    def clear(self):
        self.combined_feature_file.seek(0)
        # self.combined_id_file.seek(0)

    def load_training_data(self, negative_sample_num, matrix_type="dense"):
        query_feature_list = []
        negative_doc_feature_list = []
        positive_doc_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        query_doc_dict = {}
        query_list = []
        doc_id = 0
        for line in self.combined_feature_file:
            current_size += 1 
            if current_size > self.sample_size:
                break
            tokens = line.strip().split("\t")
            if len(tokens) < 2:
                print_log("Feature error:\t%s" % line.strip())
                continue
            query = tokens[0]
            title = tokens[1]
            query_list.append(query)
            if query not in query_doc_dict:
                query_doc_dict[query] = set()
            query_doc_dict[query].add(doc_id)
            # query_feature = transfer_feature_dict2list(get_feature_from_string_v2(query))
            query_feature = query
            for i in range(negative_sample_num):
                query_feature_list.append(query_feature)
            # positive_sample_feature = transfer_feature_dict2list(get_feature_from_string_v2(title))
            positive_sample_feature = title
            for i in range(negative_sample_num):
                positive_doc_feature_list.append(\
                    positive_sample_feature)
            doc_id += 1
        doc_size = len(positive_doc_feature_list)
        for i in range(len(query_list)):
            query = query_list[i]
            noclick_doc_id_list = []
            for j in range(negative_sample_num):
                while True:
                    negative_sample_id = random.randint(0, doc_size-1)
                    if negative_sample_id not in query_doc_dict[query]:
                        noclick_doc_id_list.append(negative_sample_id)
                        break
                    print_log("Negative sample is collision.")
            for noclick_doc_id in noclick_doc_id_list:
                negative_doc_feature_list.append(positive_doc_feature_list[noclick_doc_id])
        if matrix_type == "sparse":
            return transfer_feature_dense2sparse(query_feature_list, self.feature_num), \
                    transfer_feature_dense2sparse(positive_doc_feature_list, self.feature_num), \
                    transfer_feature_dense2sparse(negative_doc_feature_list, self.feature_num)
        elif matrix_type == "dense":
            return transfer_feature_sparse2dense(query_feature_list, self.feature_num), \
                    transfer_feature_sparse2dense(positive_doc_feature_list, self.feature_num), \
                    transfer_feature_sparse2dense(negative_doc_feature_list, self.feature_num)
        elif matrix_type == "index":
            return get_feature_index(query_feature_list, self.feature_num), \
                    get_feature_index(positive_doc_feature_list, self.feature_num), \
                    get_feature_index(negative_doc_feature_list, self.feature_num)

    def generate_training_data_v2(self, negative_sample_num):
        query_feature_list = []
        negative_doc_feature_list = []
        positive_doc_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        query_doc_dict = {}
        query_list = []
        doc_id = 0
        for line in self.combined_feature_file:
            tokens = line.strip().split("\t")
            if len(tokens) < 2:
                # print_log("Feature error:\t%s" % line.strip())
                continue
            query = tokens[0]
            title = tokens[1]
            query_list.append(query)
            if query not in query_doc_dict:
                query_doc_dict[query] = set()
            query_doc_dict[query].add(doc_id)
            query_feature = query
            for i in range(negative_sample_num):
                query_feature_list.append(query_feature)
            positive_sample_feature = title
            for i in range(negative_sample_num):
                positive_doc_feature_list.append(\
                    positive_sample_feature)
            doc_id += 1
            current_size += 1 
            if current_size % self.sample_size == 0:
                doc_size = len(positive_doc_feature_list)
                for i in range(len(query_list)):
                    query = query_list[i]
                    noclick_doc_id_list = []
                    for j in range(negative_sample_num):
                        while True:
                            negative_sample_id = random.randint(0, doc_size-1)
                            if negative_sample_id not in query_doc_dict[query]:
                                noclick_doc_id_list.append(negative_sample_id)
                                break
                    for noclick_doc_id in noclick_doc_id_list:
                        negative_doc_feature_list.append(positive_doc_feature_list[noclick_doc_id])
                query_feature = get_feature_index(query_feature_list, self.feature_num)
                positive_doc_feature = get_feature_index(positive_doc_feature_list, self.feature_num)
                negative_doc_feature = get_feature_index(negative_doc_feature_list, self.feature_num)
                for feature in query_feature:
                    for f in feature:
                        if np.isinf(f):
                            print feature
                yield ([query_feature,\
                        positive_doc_feature,\
                        negative_doc_feature],\
                        np.ones((self.sample_size, 1)))
                query_feature_list = []
                positive_doc_feature_list = []
                negative_doc_feature_list = []
                query_list = []
                doc_id = 0
                query_doc_dict = {}

    def generate_sparse_training_data(self, negative_sample_num):
        query_feature_list = []
        negative_doc_feature_list = []
        positive_doc_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        query_doc_dict = {}
        query_list = []
        doc_id = 0
        for line in self.combined_feature_file:
            tokens = line.strip().split("\t")
            if len(tokens) < 2:
                # print_log("Feature error:\t%s" % line.strip())
                continue
            query = tokens[0]
            title = tokens[1]
            query_list.append(query)
            if query not in query_doc_dict:
                query_doc_dict[query] = set()
            query_doc_dict[query].add(doc_id)
            query_feature = query
            for i in range(negative_sample_num):
                query_feature_list.append(query_feature)
            positive_sample_feature = title
            for i in range(negative_sample_num):
                positive_doc_feature_list.append(\
                    positive_sample_feature)
            doc_id += 1
            current_size += 1 
            if current_size % self.sample_size == 0:
                doc_size = len(positive_doc_feature_list)
                for i in range(len(query_list)):
                    query = query_list[i]
                    noclick_doc_id_list = []
                    for j in range(negative_sample_num):
                        while True:
                            negative_sample_id = random.randint(0, doc_size-1)
                            if negative_sample_id not in query_doc_dict[query]:
                                noclick_doc_id_list.append(negative_sample_id)
                                break
                    for noclick_doc_id in noclick_doc_id_list:
                        negative_doc_feature_list.append(positive_doc_feature_list[noclick_doc_id])
                query_feature = transfer_feature_dense2sparse(query_feature_list, self.feature_num)
                positive_doc_feature = transfer_feature_dense2sparse(positive_doc_feature_list, self.feature_num)
                negative_doc_feature = transfer_feature_dense2sparse(negative_doc_feature_list, self.feature_num)
                yield ([query_feature,\
                        positive_doc_feature,\
                        negative_doc_feature],\
                        np.ones((self.sample_size, 1)))
                query_feature_list = []
                positive_doc_feature_list = []
                negative_doc_feature_list = []
                query_list = []
                doc_id = 0
                query_doc_dict = {}

    def generate_dense_training_data(self, negative_sample_num):
        query_feature_list = []
        negative_doc_feature_list = []
        positive_doc_feature_list = []
        current_size = 0
        print_log("Loading combined data")
        query_doc_dict = {}
        query_list = []
        doc_id = 0
        for line in self.combined_feature_file:
            tokens = line.strip().split("\t")
            if len(tokens) < 2:
                # print_log("Feature error:\t%s" % line.strip())
                continue
            query = tokens[0]
            title = tokens[1]
            query_list.append(query)
            if query not in query_doc_dict:
                query_doc_dict[query] = set()
            query_doc_dict[query].add(doc_id)
            query_feature = query
            for i in range(negative_sample_num):
                query_feature_list.append(query_feature)
            positive_sample_feature = title
            for i in range(negative_sample_num):
                positive_doc_feature_list.append(\
                    positive_sample_feature)
            doc_id += 1
            current_size += 1 
            if current_size % self.sample_size == 0:
                doc_size = len(positive_doc_feature_list)
                for i in range(len(query_list)):
                    query = query_list[i]
                    noclick_doc_id_list = []
                    for j in range(negative_sample_num):
                        while True:
                            negative_sample_id = random.randint(0, doc_size-1)
                            if negative_sample_id not in query_doc_dict[query]:
                                noclick_doc_id_list.append(negative_sample_id)
                                break
                    for noclick_doc_id in noclick_doc_id_list:
                        negative_doc_feature_list.append(positive_doc_feature_list[noclick_doc_id])
                query_feature = transfer_feature_sparse2dense(query_feature_list, self.feature_num)
                positive_doc_feature = transfer_feature_sparse2dense(positive_doc_feature_list, self.feature_num)
                negative_doc_feature = transfer_feature_sparse2dense(negative_doc_feature_list, self.feature_num)
                for feature in query_feature:
                    for f in feature:
                        if np.isinf(f):
                            print feature
                yield ([query_feature,\
                        positive_doc_feature,\
                        negative_doc_feature],\
                        np.ones((self.sample_size, 1)))
                query_feature_list = []
                positive_doc_feature_list = []
                negative_doc_feature_list = []
                query_list = []
                doc_id = 0
                query_doc_dict = {}

if __name__ == "__main__":
    # Prepare for theano
    training_data = TrainingData()
    cProfile.run("training_data.load_sparse_training_data_v3()")
