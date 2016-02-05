import operator
import math

from flask import Flask, render_template, request
from flask.helpers import url_for

import sys
reload(sys)
sys.setdefaultencoding('utf8')

QID_KEY = "m:QueryId"
QUERY_KEY = "m:Query"
URL_KEY = "m:Url"
LABEL_KEY = "m:Rating"
DOCID_KEY = "m:DocId"
RANK_KEY = "FN_FINAL_RANK"

class FeatureDict(object):
    def __init__(self, raw_file_path, feature_file_path, featuare_database_path):
        self.feature_dict = {}
        self.head_set = ["detail"]
        for line in open(raw_file_path):
            tokens = line.strip().split("\t")
            query = tokens[0].strip("[]")
            url_md5 = tokens[1].strip("[]").split(":")[0]
            title = tokens[3]
            content = tokens[4]
            detail = " ".join(tokens[5:])
            if query in self.feature_dict:
                self.feature_dict[query][url_md5] = {}
                self.feature_dict[query][url_md5]["title"] = title
                self.feature_dict[query][url_md5]["content"] = content
                self.feature_dict[query][url_md5]["detail"] = detail 
            else:
                self.feature_dict[query] = {url_md5:{}}
                self.feature_dict[query][url_md5]["title"] = title
                self.feature_dict[query][url_md5]["content"] = content
                self.feature_dict[query][url_md5]["detail"] = detail 

        feature_file = open(feature_file_path)
        for line in feature_file:
            self.head_set = line.strip().split("\t")
            break
        for line in feature_file:
            tokens = line.strip().split("\t")
            sample_feature_dict = {}
            for i in xrange(len(tokens)):
                if i == 0 or i > 5:
                    sample_feature_dict[self.head_set[i]] = float(tokens[i])
                else:
                    sample_feature_dict[self.head_set[i]] = tokens[i]
            query = sample_feature_dict[QUERY_KEY]
            url_md5 = sample_feature_dict[URL_KEY][16:]
            if query in self.feature_dict:
                if url_md5 in self.feature_dict[query]:
                    for key in sample_feature_dict:
                        self.feature_dict[query][url_md5][key.split(":")[-1]] = sample_feature_dict[key]

    def sort_data(data, key):
        for qid in data:
            data[qid].sort(key=lambda x: float(x[key]), reverse=True)

    def get_search_results(self, query, score):
        if query not in self.feature_dict:
            return []
        return sorted(self.feature_dict[query].values(), key=lambda x: float(x[score]), reverse=True)

    def calc_dcg(data, n, dcg_set):
        dcg = 0
        qn = 0
        for qid in data:
            d = data[qid]
            cur_dcg = 0
            for i in xrange(0, n):
                if len(d) <= i:
                    break
                #print d[i]["m:Url"]
                cur_dcg += (math.pow(2, d[i][LABEL_KEY]) - 1)/(math.log(i + 2))
            dcg += cur_dcg
            dcg_set[qid] = cur_dcg
            qn += 1
        return dcg/qn

    def is_better_than(gdd, d):
        if gdd["FN_HIT_TITLE_IDF"] < d["FN_HIT_TITLE_IDF"]:
            return False
        if gdd["FN_HIT_TITLE_MISS_CTW"] > d["FN_HIT_TITLE_MISS_CTW"]:
            return False
        if d["FN_DOC_PV_NUM"] > 10 and gdd["FN_DOC_PV_NUM"] < d["FN_DOC_PV_NUM"]:
            return False
        return True

    def semantic_similarity_rank(data, similarity_metric):
        data.sort(key=lambda x:x[similarity_metric], reverse=True)
        max_score = len(data) + 1
        for doc in data:
            doc[RANK_KEY] = max_score
            max_score -= 1

    def qp_site_rank(data):
        in_domain_num = 0
        total = len(data)
        good_domain_data = []
        for i in xrange(0, total):
            d = data[i]
            if d["FN_DOMAIN_IN_QUERY_PATTERN_SET"] > 0:
                in_domain_num += 1
                good_domain_data.append(d)
        #print in_domain_num, total
        if float(in_domain_num) / total > 0.5 or True:
            in_domain_index = 0
            i = 0
            while i < total and in_domain_index < in_domain_num:
                d = data[i]
                gdd = good_domain_data[in_domain_index]
                #print d[DOCID_KEY], gdd[DOCID_KEY]
                if d[DOCID_KEY] == gdd[DOCID_KEY]:
                    i += 1
                    in_domain_index += 1
                    continue
                if d["FN_IS_INDEX_DOC"] > 0 or d["FN_IS_HOME_PAGE"] > 0:
                    i += 1
                    continue
                if is_better_than(gdd, d):
                    gdd[RANK_KEY] = d[RANK_KEY] + 1
                    in_domain_index += 1
                    if gdd["m:Rating"] >= d["m:Rating"]:
                        print "good case:%s %s replace %s" % (gdd["m:Query"], gdd["m:UrlValue"], d["m:UrlValue"])
                    else:
                        print "bad case:%s %s replace %s" % (gdd["m:Query"], gdd["m:UrlValue"], d["m:UrlValue"])
                else:
                    i += 1

    def rerank(file):
        data = load_data(file)
        sort_data(data, RANK_KEY)
        ori_dcg_set = {}
        ori_dcg = calc_dcg(data, 3, ori_dcg_set)
        for qid in data:
            # qp_site_rank(data[qid])
            semantic_similarity_rank(data[qid], "FN_QUERY_TITLE_SEMANTIC_SIMALARITY")
        new_dcg_set = {}
        sort_data(data, RANK_KEY)
        new_dcg = calc_dcg(data, 3, new_dcg_set)

        good_num = 0
        bad_num = 0
        same_num = 0
        for qid in ori_dcg_set:
            if qid in new_dcg_set:
                if new_dcg_set[qid] > ori_dcg_set[qid] + 0.01:
                    good_num += 1
                    print "good_query %s" % (data[qid][0]["m:Query"])
                elif new_dcg_set[qid] < ori_dcg_set[qid]:
                    print "bad_query %s" % (data[qid][0]["m:Query"])
                    bad_num += 1
                else:
                    same_num += 1
        print "ori_dcg:%f, new_dcg:%f, good:%d, bad:%d, same:%d" % (ori_dcg, new_dcg, good_num, bad_num, same_num)

feature_dict = FeatureDict("../data/feature.raw.test", "../data/feature.extract.test", "database/feature_dict")
app = Flask(__name__)

@app.route("/search", methods=['POST', 'GET'])
def search():
    selected_feature_set = set()
    query = request.args.get('q').encode("utf-8")
    for feature_name in feature_dict.head_set:
        if request.args.get(feature_name, False):
            print feature_name
            selected_feature_set.add(feature_name)
    results_a = feature_dict.get_search_results(query, "FN_FINAL_RANK")
    results_b = feature_dict.get_search_results(query, "FN_QUERY_TITLE_SEMANTIC_SIMALARITY")
    count = request.args.get('count')
    if not count:
        count = 10
    else:
        count = int(count)
    return render_template('search.html', query=query, results_a=results_a, results_b=results_b, query_list=feature_dict.feature_dict.keys(), feature_list=feature_dict.head_set, selected_feature_set=selected_feature_set)

if __name__ == '__main__':
    app.run(debug=True,  host='0.0.0.0', port=7778)
