import sys
sys.path.append("/home/search/tangjie/dssm/src/")
import operator
import math
import urllib2
import re
import theano
import scipy.sparse as sp
import cPickle
import time

from flask import Flask, render_template, request
from flask.helpers import url_for

reload(sys)
sys.setdefaultencoding('utf8')

from util.util import *
from training.dssm import *
from training.training_data import *

def compute_cosine_between_matrixes(q_matrix, d_matrix):
    qd_dot_vector = T.sum(q_matrix * d_matrix, 1)
    q_norm_vector = T.sqrt(T.sum(T.sqr(q_matrix), 1) + EPSILON)
    d_norm_vector = T.sqrt(T.sum(T.sqr(d_matrix), 1) + EPSILON)
    return qd_dot_vector / (q_norm_vector * d_norm_vector)

def get_original_doc_list(query):
    url = "http://10.138.240.99:8360/debug.htm?&rewrite=1&dbg=1&cache=0&param=sep_fuzzy:1|sep_synm:1&kw=%s&count=100" % query
    headers = {  
        'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'  
    }  
    req = urllib2.Request(  
        url = "http://10.138.240.99:8360/debug.htm?&rewrite=1&dbg=1&cache=0&param=sep_fuzzy:1|sep_synm:1&kw=%s&count=100" % query,
        headers = headers
    )
    response = urllib2.urlopen(req) 
    html = response.read()
    p_title = re.compile("<br><a.*?href=\"(.+?)\".*?>(.+?)</a>")
    m_title_list = p_title.finditer(html)
    result_list = []
    url_set = set()
    for m_title in m_title_list:
        # print_log("%s" % m_title.group())
        result_info = {}
        url = m_title.group(1)
        if url in url_set:
            continue
        url_set.add(url)
        result_info["url"] = url
        title = m_title.group(2).replace("<b>", "").replace("</b>", "").replace(" ", "")
        try:
            result_info["title"] = segment_sentence(title)
        except:
            print_log("[ERROR SEGMENT] %s", title)
            continue
        result_list.append(result_info)
    return result_list

def segment_sentence(sentence):
    url = "http://10.138.83.177:8360/qrwt.htm?&kw=%s" % sentence
    headers = {  
        'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'  
    }  
    req = urllib2.Request(  
        url = "http://10.172.173.238:8360/qrwt.htm?&kw=%s" % sentence,
        headers = headers
    )
    response = urllib2.urlopen(req) 
    html = response.read()
    p_phrase_list = re.compile("<table.+?table>.+?<table border=\"1\">(.+?)</table>")
    m_phrase_list = p_phrase_list.search(html)
    token_list = ""
    if m_phrase_list:
        phrase_list = m_phrase_list.group(1)
        p_token_list = re.compile("<tr><td>.*?</td><td>.*?</td><td>.*?</td><td>(.*?)</td></tr>")
        m_token_list = p_token_list.finditer(phrase_list)
        for m_token in m_token_list:
            token_list += m_token.group(1)
    return token_list.strip("\[\]").replace("][", "\x01")

def calculate_cosine_similarity(h_wds, feature_num, query, result_list, test_function, test_q, test_d):
    print_log("Starting predict")
    query_feature_list = []
    title_feature_list = []
    print_log("%s" % query)
    query_feature_string = generate_sentence_feature_v2(h_wds, query)
    # print_log("%s" % query_feature_string)
    if not query_feature_string:
        return []
    query_feature = get_feature_from_string(query_feature_string)
    for result in result_list:
        title_feature_string = generate_sentence_feature_v2(h_wds, result["title"])
        if not title_feature_string:
            title_feature_string = "0:1"
        query_feature_list.append(query_feature)
        title_feature_list.append(\
                get_feature_from_string(title_feature_string))
    # print_log("%s" % query_feature_list)
    # print_log("%s" % title_feature_list)
    test_q.set_value(\
            transfer_feature_dense2sparse(\
                    query_feature_list, feature_num))
    test_d.set_value(\
            transfer_feature_dense2sparse(\
                    title_feature_list, feature_num))
    cosine_similarity_list = test_function()
    # print_log("%s" % cosine_similarity_list)
    return cosine_similarity_list

def rerank(query, result_list):
    for result in result_list:
        title = result["title"]


app = Flask(__name__)

@app.route("/search", methods=['POST', 'GET'])
def search():
    query = request.args.get('q').encode("utf-8").replace(" ", "")
    result_list = get_original_doc_list(query)
    segmented_query = segment_sentence(query)
    # results_b = get_original_doc_list(query)
    count = request.args.get('count')
    cosine_similarity_list = calculate_cosine_similarity(h_wds, feature_num, segmented_query, result_list, test_function, test_q, test_d)
    for i in range(len(cosine_similarity_list)):
        result_list[i]["cosine_similarity"] = cosine_similarity_list[i]
    return render_template('search.html', query=query, results_a=result_list, results_b=result_list)

if __name__ == '__main__':
    print_log("Loading word hash dict")
    dic_file = open('/home/search/tangjie/dssm/data/word_hash_dict2_100', 'r')
    h_wds = loadHashDictionary(dic_file)

    feature_num = 62325
    negative_d_num = 4
    mini_batch_size = 1
    test_q = theano.shared(sp.csr_matrix((mini_batch_size, feature_num), dtype=theano.config.floatX))
    test_d = theano.shared(sp.csr_matrix(((negative_d_num + 1) *mini_batch_size, feature_num), dtype=theano.config.floatX))

    print_log("Loading dssm model")
    model_file_name = "../../model/dssm_gamma_4096_0.010000_0_763.save"
    net = cPickle.load(open(model_file_name, "rb"))
    # test = theano.function(inputs=[], outputs=net.cosine_layer.test(),\
    #                     givens = {net.q:test_q, net.d:test_d})
    test_function = theano.function(inputs=[], \
            outputs=compute_cosine_between_matrixes(net.cosine_layer.q, net.cosine_layer.d), givens={net.q:test_q, net.d:test_d})

    app.run(debug=True,  host='0.0.0.0', port=7777)
