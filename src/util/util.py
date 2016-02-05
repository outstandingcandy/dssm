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

def generate_sentence_feature(h_wds, sentence):
    q_to_h = {}
    q_unsat = False
    sentence = sentence.decode("utf-8")
    for term in sentence.split('\1'):
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
                print 'key is not contained in query', str(idx) + term[idx]
                q_unsat = True
                break
        if q_unsat:
            break
    if len(q_to_h) and not q_unsat:
        return dicToString(q_to_h)

def generate_sentence_feature_v2(h_wds, sentence):
    q_to_h = {}
    q_pos = []
    q_unsat = False
    sentence = sentence.decode("utf-8")
    for term in sentence.split('\1'):
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


