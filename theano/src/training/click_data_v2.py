import sys
import random
import cPickle
import theano
import numpy as np
from util import *
import scipy.sparse as sp

GOOD_TITLE_NUM = 1
BAD_TITLE_NUM = 4

class ClickData(object):

    def __init__(self, from_raw_file=True, \
            query_feature_file_name="../data/query_feature", doc_feature_file_name="../data/doc_feature",\
            click_data_file_name="../data/aux_file"):
        self.query_feature_file = open(query_feature_file_name)
        self.doc_feature_file = open(doc_feature_file_name)
        self.click_data_file = open(click_data_file_name)
        self.last_click_data_file_cursor = 0
        self.current_doc_id = 1
        self.using_noclick_data = False
        self.negative_sample_num = 4
        print_log("Get click data.")
        self.query_doc_pair, self.query_doc_dict = self.get_query_doc_pair_v2(self.click_data_file, 100000000)
        if from_raw_file:
            print_log("Get query feature.")
            self.query_feature_list = self.get_query_feature_string(self.query_feature_file)
            query_feature_dump_file = open("%s.cPickle" % (query_feature_file_name), "wb")
            cPickle.dump(self.query_feature_list, query_feature_dump_file)
            query_feature_dump_file.close()
        else:
            print_log("Get query feature from dump file.")
            query_feature_dump_file = open("%s.cPickle" % (query_feature_file_name), "rb")
            self.query_feature_list = cPickle.load(query_feature_dump_file)
            query_feature_dump_file.close()

    def get_query_feature_string(self, query_feature_file):
        feature_list = []
        for line in query_feature_file:
            feature_string = line.strip().replace("\t", " ")
            feature_list.append(feature_string)
        return feature_list

    def get_doc_feature_string(self, end_doc_id):
        doc_list_feature = {}
        for line in self.doc_feature_file:
            feature_string = ""
            tokens = line.strip().split("\t")
            for token in tokens[:-3]:
                feature_string += token + " "
            doc_list_feature[self.current_doc_id] = {}
            doc_list_feature[self.current_doc_id]["feature"] = feature_string.strip()
            doc_list_feature[self.current_doc_id]["impression"] = int(tokens[-3])
            doc_list_feature[self.current_doc_id]["examination"] = int(tokens[-2])
            doc_list_feature[self.current_doc_id]["click"] = int(tokens[-1])
            self.current_doc_id += 1
            if self.current_doc_id > end_doc_id:
                print self.current_doc_id
                print line.strip()
                break
        return doc_list_feature

    def get_query_doc_pair(self, feature_file_name):
        click_pair_list = []
        for line in feature_file_name:
            tokens = line.split("\t")
            query_id = int(tokens[0])
            begin_doc_id = int(tokens[1])
            end_doc_id = int(tokens[2])
            click_pair_list.append((query_id, begin_doc_id, end_doc_id))
        return click_pair_list

    def get_query_doc_pair_v2(self, feature_file_name, positive_sample_num):
        click_pair_list = []
        doc_id = 0
        new_query_id = 0 
        query_id_set = set()
        for line in feature_file_name:
            tokens = line.split("\t")
            query_id = int(tokens[0])
            if query_id not in query_id_set:
                new_query_id += 1
                query_id_set.add(query_id)
            doc_id += 1
            if doc_id > positive_sample_num:
                break
            begin_doc_id = doc_id
            end_doc_id = doc_id
            click_pair_list.append((new_query_id, begin_doc_id, end_doc_id))
        click_pair_dict = {}
        for query_id, begin_doc_id, end_doc_id in click_pair_list:
            if query_id not in click_pair_dict:
                click_pair_dict[query_id] = set()
            click_pair_dict[query_id].add(begin_doc_id)
        return click_pair_list, click_pair_dict

    def dump_query_doc_pair_feature(self, begin_query_id, end_query_id, combined_feature_file, combined_id_file):
        click_feature_dict = {}
        GOOD_TITLE_ID = 0
        BAD_TITLE_ID = 1
        qid_count = 0
        total_begin_doc_id = 100000000000000
        total_end_doc_id = 0
        current_data_size = 0
        print_log("Find the docid range.")
        for (query_id, begin_doc_id, end_doc_id) in self.query_doc_pair[begin_query_id : end_query_id]:
            if begin_doc_id < total_begin_doc_id:
                total_begin_doc_id = begin_doc_id
            if end_doc_id > total_end_doc_id:
                total_end_doc_id = end_doc_id
        print_log("docid range:\t%d\t%d" % (total_begin_doc_id, total_end_doc_id))
        if total_begin_doc_id < self.current_doc_id:
            print_log("Get doc feature error")
        else:
            doc_list_feature = self.get_doc_feature_string(total_end_doc_id)
        print_log("Get positive and negative samples")
        for (query_id, begin_doc_id, end_doc_id) in self.query_doc_pair[begin_query_id : end_query_id]:
            if query_id not in click_feature_dict:
                click_feature_dict[query_id] = [[], []]
            for doc_id in range(begin_doc_id, end_doc_id+1):
                if doc_list_feature[doc_id]["click"] > 0:
                    click_feature_dict[query_id][GOOD_TITLE_ID].append(doc_id)
                else:
                    click_feature_dict[query_id][BAD_TITLE_ID].append(doc_id)
        print_log("Get all features")
        for query_id in click_feature_dict:
            query_feature = self.query_feature_list[query_id-1]
            for click_doc_id in click_feature_dict[query_id][GOOD_TITLE_ID]:
                combined_feature_file.write("%s" % (query_feature))
                combined_id_file.write("%d" % (query_id))
                click_doc_feature = doc_list_feature[click_doc_id]["feature"]
                combined_feature_file.write("\t%s" % (click_doc_feature))
                combined_id_file.write("\t%d" % (click_doc_id))
                noclick_doc_id_list = []
                if self.using_noclick_data:
                    for i in range(self.negative_sample_num):
                        try:
                            noclick_doc_id_list.append(random.choice(click_feature_dict[query_id][BAD_TITLE_ID]))
                        except:
                            print_log("query_id:%d has no bad title" % (query_id))
                            break
                else:
                    for i in range(self.negative_sample_num):
                        noclick_doc_id_list.append(random.randint(total_begin_doc_id, total_end_doc_id))
                if len(noclick_doc_id_list) != self.negative_sample_num:
                    print_log("query_id:%d do not has enough negative sample" % (query_id))
                    break
                for noclick_doc_id in noclick_doc_id_list:
                    noclick_doc_feature = doc_list_feature[noclick_doc_id]["feature"]
                    combined_feature_file.write("\t%s" % (noclick_doc_feature))
                    combined_id_file.write("\t%d" % (noclick_doc_id))
                combined_feature_file.write("\n")
                combined_id_file.write("\n")

    def dump_query_doc_pair_feature_v2(self, begin_positive_sample_id, end_positive_sample_id, combined_feature_file, combined_id_file):
        click_feature_dict = {}
        GOOD_TITLE_ID = 0
        BAD_TITLE_ID = 1
        qid_count = 0
        total_begin_doc_id = 100000000000000
        total_end_doc_id = 0
        current_data_size = 0
        print_log("Find the docid range.")
        for (query_id, begin_doc_id, end_doc_id) in self.query_doc_pair[begin_positive_sample_id : end_positive_sample_id]:
            if begin_doc_id < total_begin_doc_id:
                total_begin_doc_id = begin_doc_id
            if end_doc_id > total_end_doc_id:
                total_end_doc_id = end_doc_id
        print_log("docid range:\t%d\t%d" % (total_begin_doc_id, total_end_doc_id))
        if total_begin_doc_id < self.current_doc_id:
            print_log("Get doc feature error")
        else:
            doc_list_feature = self.get_doc_feature_string(total_end_doc_id)
        print_log("Get positive samples")
        for (query_id, begin_doc_id, end_doc_id) in self.query_doc_pair[begin_positive_sample_id : end_positive_sample_id]:
            if query_id not in click_feature_dict:
                click_feature_dict[query_id] = [[], []]
            for doc_id in range(begin_doc_id, end_doc_id+1):
                if doc_list_feature[doc_id]["click"] > 0:
                    click_feature_dict[query_id][GOOD_TITLE_ID].append(doc_id)
                else:
                    print_log("Error positive samples")
        print_log("Get all features")
        for query_id in click_feature_dict:
            query_feature = self.query_feature_list[query_id-1]
            for click_doc_id in click_feature_dict[query_id][GOOD_TITLE_ID]:
                combined_feature_file.write("%s" % (query_feature))
                combined_id_file.write("%d" % (query_id))
                click_doc_feature = doc_list_feature[click_doc_id]["feature"]
                combined_feature_file.write("\t%s" % (click_doc_feature))
                combined_id_file.write("\t%d" % (click_doc_id))
                noclick_doc_id_list = []
                for i in range(self.negative_sample_num):
                    while True:
                        negative_sample_id = random.randint(total_begin_doc_id, total_end_doc_id)
                        if negative_sample_id not in self.query_doc_dict[query_id]:
                            noclick_doc_id_list.append(random.randint(total_begin_doc_id, total_end_doc_id))
                            # noclick_doc_id_list.append(random.randint(total_begin_doc_id, total_begin_doc_id+1024))
                            break
                        print_log("Negative sample is collision.")
                if len(noclick_doc_id_list) != self.negative_sample_num:
                    print_log("query_id:%d do not has enough negative sample" % (query_id))
                    break
                for noclick_doc_id in noclick_doc_id_list:
                    noclick_doc_feature = doc_list_feature[noclick_doc_id]["feature"]
                    combined_feature_file.write("\t%s" % (noclick_doc_feature))
                    combined_id_file.write("\t%d" % (noclick_doc_id))
                combined_feature_file.write("\n")
                combined_id_file.write("\n")

if __name__ == "__main__":
    # Get combined feature
    combined_feature_file = open("../data/combined_feature_50m_newhash", "w")
    combined_id_file = open("../data/combined_id_50m_newhash", "w")
    click_data = ClickData(True, "../data/all_pos_data_qry_newhash", "../data/all_pos_data_doc_newhash", "../data/all_pos_data_aux_newhash")
    for i in range(5):
        click_data.dump_query_doc_pair_feature_v2(i*10000000, (i+1)*10000000, combined_feature_file, combined_id_file)
    combined_feature_file.close()
    combined_id_file.close()

