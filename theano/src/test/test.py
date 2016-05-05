import sys
sys.path.append("/home/search/tangjie/dssm/src/")
import os
import cPickle

import scipy.sparse as sp
import theano
from theano import sparse as TS

from util.util import *
from training.dssm import *
from training.training_data import *
from util.chinese import *

theano.config.floatX = 'float32'
theano.config.on_unused_input = 'warn'

def compute_cosine_between_matrixes(q_matrix, d_matrix):
    qd_dot_vector = T.sum(q_matrix * d_matrix, 1)
    q_norm_vector = T.sqrt(T.sum(T.sqr(q_matrix), 1) + EPSILON)
    d_norm_vector = T.sqrt(T.sum(T.sqr(d_matrix), 1) + EPSILON)
    return qd_dot_vector / (q_norm_vector * d_norm_vector)

def test(testing_data_file_name="../data/combined_feature", sample_size=5, model_file_name="../model/dssm_alpha_6_105.save"):
    print_log("Loading word hash dict")
    dic_file = open('/home/search/tangjie/dssm/data/word_hash_dict2_100', 'r')
    h_wds = loadHashDictionary(dic_file)

    feature_num = 62325
    negative_d_num = 4
    mini_batch_size = 1
    test_q = theano.shared(sp.csr_matrix((mini_batch_size, feature_num), dtype=theano.config.floatX))
    test_d = theano.shared(sp.csr_matrix(((negative_d_num + 1) *mini_batch_size, feature_num), dtype=theano.config.floatX))

    print_log("Loading dssm model")
    net = cPickle.load(open(model_file_name, "rb"))
    # test = theano.function(inputs=[], outputs=net.cosine_layer.test(),\
    #                     givens = {net.q:test_q, net.d:test_d})
    test = theano.function(inputs=[], \
            outputs=compute_cosine_between_matrixes(net.cosine_layer.q, net.cosine_layer.d), givens={net.q:test_q, net.d:test_d})

    print_log("Starting predict")
    for line in sys.stdin:
        line = line.strip()
        tokens = line.split("\t")
        if len(tokens) == 2:
            query_feature_list = []
            title_feature_list = []
            query = tokens[0]
            title = tokens[1]
            query_feature_string = generate_sentence_feature_v2(h_wds, query)
            title_feature_string = generate_sentence_feature_v2(h_wds, title)
            if not (query_feature_string and title_feature_string):
                print_log("feature error")
                continue
            query_feature_list.append(\
                    get_feature_from_string(query_feature_string))
            title_feature_list.append(\
                    get_feature_from_string(title_feature_string))
            # print_log("%s" % query_feature_list)
            # print_log("%s" % title_feature_list)
            test_q.set_value(\
                    transfer_feature_dense2sparse(query_feature_list, feature_num))
            test_d.set_value(\
                    transfer_feature_dense2sparse(title_feature_list, feature_num))
            print query, title, " ".join(tokens[2:]), test()
        elif len(tokens) > 2:
            query_feature_list = []
            title_feature_list = []
            query = tokens[2]
            title = tokens[4]
            url_md5, label = tokens[3].split(" ")
            query_feature_string = generate_sentence_feature_v2(h_wds, query)
            title_feature_string = generate_sentence_feature_v2(h_wds, title)
            if not (query_feature_string and title_feature_string):
                print_log("feature error")
                continue
            query_feature_list.append(\
                    get_feature_from_string(query_feature_string))
            title_feature_list.append(\
                    get_feature_from_string(title_feature_string))
            # print_log("%s" % query_feature_list)
            # print_log("%s" % title_feature_list)
            test_q.set_value(\
                    transfer_feature_dense2sparse(query_feature_list, feature_num))
            test_d.set_value(\
                    transfer_feature_dense2sparse(title_feature_list, feature_num))
            cos_similarity = test()[0]
            print "%s\t%s\t%f\t%f\t%s\t-1\t%s\t%s\t%s" % \
                    (tokens[0], tokens[1], cos_similarity, cos_similarity, label, query.replace("\x01", ""), url_md5, title.replace("\x01", ""))

def test_v2(testing_data_file_name="../data/combined_feature", sample_size=5, model_file_name="../model/dssm_alpha_6_105.save"):
    print_log("Loading word hash dict")
    dic_file = open('/home/search/tangjie/dssm/data/word_hash_dict2_100', 'r')
    h_wds = loadHashDictionary(dic_file)

    feature_num = 62325
    negative_d_num = 4
    mini_batch_size = 1
    test_q = theano.shared(sp.csr_matrix((mini_batch_size, feature_num), dtype=theano.config.floatX))
    test_d = theano.shared(sp.csr_matrix(((negative_d_num + 1) *mini_batch_size, feature_num), dtype=theano.config.floatX))

    print_log("Loading dssm model")
    net = cPickle.load(open(model_file_name, "rb"))
    # test = theano.function(inputs=[], outputs=net.cosine_layer.test(),\
    #                     givens = {net.q:test_q, net.d:test_d})
    test = theano.function(inputs=[], \
            outputs=compute_cosine_between_matrixes(net.cosine_layer.q, net.cosine_layer.d), givens={net.q:test_q, net.d:test_d})

    print_log("Starting predict")
    query_feature_list = []
    title_feature_list = []
    line_list = []
    for line in sys.stdin:
        line = line.strip()
        tokens = line.split("\t")
        if len(tokens) > 2:
            query = tokens[2]
            title = tokens[4]
            url_md5, label = tokens[3].split(" ")
            query_feature_string = generate_sentence_feature_v2(h_wds, query)
            title_feature_string = generate_sentence_feature_v2(h_wds, title)
            if not (query_feature_string and title_feature_string):
                print_log("feature error")
                continue
            query_feature_list.append(\
                    get_feature_from_string(query_feature_string))
            title_feature_list.append(\
                    get_feature_from_string(title_feature_string))
            # print_log("%s" % query_feature_list)
            # print_log("%s" % title_feature_list)
            line_list.append(line)
        else:
            print_log(line)

    test_q.set_value(\
            transfer_feature_dense2sparse(query_feature_list, feature_num))
    test_d.set_value(\
            transfer_feature_dense2sparse(title_feature_list, feature_num))
    cos_similarity_list = test()
    for i in range(len(line_list)):
        tokens = line_list[i].split("\t")
        query = tokens[2]
        title = tokens[4]
        url_md5, label = tokens[3].split(" ")
        print "%s\t%s\t%f\t%f\t%s\t-1\t%s\t%s\t%s" % \
                (tokens[0], tokens[1], cos_similarity_list[i], cos_similarity_list[i], label, query.replace("\x01", ""), url_md5, title.replace("\x01", ""))

if __name__ == "__main__":
    os.chdir("/home/search/tangjie/dssm/src")
    test(model_file_name="../model/dssm_alpha_4096_0.100000_2_2289.save")
