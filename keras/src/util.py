import sys
from datetime import datetime
from chinese import *

def print_log(log):
    sys.stderr.write("%s\t%s\n" % (str(datetime.now()), log))

def loadHashDictionary(fn):
    h_wds = {}
    for line in fn:
        segs = line.strip().decode('utf-8').split('\t')
        h_wds[segs[0]] = int(segs[1])
    return h_wds

def getAlphaNumFeature(term, h_wds, q_to_h, q_pos):
    seq = '#' + term + '#'
    for i in xrange(len(seq) -2):
        key = seq[i:i+3]
        if h_wds.has_key(key):
            pos = h_wds[key]
            if q_to_h.has_key(pos):
                q_to_h[pos] += 1
            else:
                q_to_h[pos] = 1
            q_pos.append(key)

def dicToString(h_wds, dic, key_pos):
    feat = ''
    for key in key_pos:
        pos = h_wds[key]
        feat += str(pos) + ':' + str(dic[pos]) + ' '
    return feat[:-1]

def generate_sentence_feature(h_wds, sentence, split = '\1'):
    q_to_h = {}
    q_unsat = False
    sentence = sentence.decode("utf-8")
    for term in sentence.split(split):
        term = term.strip()
        for idx in xrange(len(term)):
            key = str(idx)+term[idx]
            if h_wds.has_key(key):
                pos = h_wds[key]
                if q_to_h.has_key(pos):
                    q_to_h[pos] += 1
                else:
                    q_to_h[pos] = 1
            else: # error!
                # print 'key is not contained in query', str(idx) + term[idx]
                q_unsat = True
                break
        if q_unsat:
            break
    if len(q_to_h) and not q_unsat:
        return dicToString(q_to_h)

def init_stop_dict(stop_word_path):
    stop_word_set = set()
    for line in open(stop_word_path,"r"):
        line = line.strip()
        stop_word_set.add(line)
    return stop_word_set

def filter_sentence(stop_word_set, sentence, split = '\1'):
    new_sentence = ""
    terms = sentence.strip().split(split)
    for term in terms:
        if is_other_all(term.decode("utf-8","ignore")):
            continue
        if term in stop_word_set:
            continue
        new_sentence += term + split
    return new_sentence.strip(split)

def generate_sentence_feature_v2(h_wds, sentence, split = '\1'):
    q_to_h = {}
    q_pos = []
    q_unsat = False
    sentence = sentence.decode("utf-8").lower()
    for term in sentence.split(split):
        term = term.strip()
        if is_alpha_or_num_all(term):
            getAlphaNumFeature(term, h_wds, q_to_h, q_pos)
            continue
        for idx in xrange(len(term)):
            key = str(idx)+term[idx]
            if h_wds.has_key(key):
                pos = h_wds[key]
                if q_to_h.has_key(pos):
                    q_to_h[pos] += 1
                else:
                    q_to_h[pos] = 1
                q_pos.append(key)
            # else: # error!
                # print 'key is not contained in query', str(idx) + term[idx]
                # q_unsat = True
                # return ""
        if q_unsat:
            break
    if len(q_to_h) and not q_unsat:
        return dicToString(h_wds, q_to_h, q_pos)
    else:
        print_log("ERROR:\t%s" % sentence.encode("utf-8", "ignore"))
        return ""

def generate_sentence_term_feature(h_wds, sentence, split = '\1'):
    q_to_h = {}
    q_pos = []
    sentence = sentence.decode("utf-8").lower()
    for term in sentence.split(split):
        term = term.strip()
        if is_alpha_or_num_all(term):
            getAlphaNumFeature(term, h_wds, q_to_h, q_pos)
            continue
        key = term
        if h_wds.has_key(key):
            pos = h_wds[key]
            if q_to_h.has_key(pos):
                q_to_h[pos] += 1
            else:
                q_to_h[pos] = 1
            q_pos.append(key)
    if len(q_to_h):
        return dicToString(h_wds, q_to_h, q_pos)
    else:
        print_log("ERROR:\t%s" % sentence.encode("utf-8", "ignore"))
        return ""

def generate_sentence_term_feature_v2(h_wds, sentence, split = '\1'):
    q_to_h = {}
    q_pos = []
    sentence = sentence.decode("utf-8").lower()
    for term in sentence.split(split):
        term = term.strip()
        key = term
        if h_wds.has_key(key):
            pos = h_wds[key]
            if q_to_h.has_key(pos):
                q_to_h[pos] += 1
            else:
                q_to_h[pos] = 1
            q_pos.append(key)
    if len(q_to_h):
        return dicToString(h_wds, q_to_h, q_pos)
    else:
        print_log("ERROR:\t%s" % sentence.encode("utf-8", "ignore"))
        return ""
