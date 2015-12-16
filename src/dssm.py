import theano
import numpy as np
from theano import tensor as T
from theano.tensor import tanh
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

GOOD_TITLE_NUM = 10
BAD_TITLE_NUM = 4
EPSILON = 0.0000000001

GPU = False
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True."

theano.config.floatX = 'float32'
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'

def size(data):
    "Return the size of the dataset `data`."
    #return data[0].get_value(borrow=True).shape[0]
    return data.get_value(borrow=True).shape[0]

class FullyConnectedLayer(object):
    
    def __init__(self, n_in, n_out, activation_fn=tanh):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)

        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)

        self.params = [self.w, self.b]

    def set_inpt(self, inpts_q, inpts_t1, inpts_t2, mini_batch_size):
        self.inpt_q = inpts_q.reshape((mini_batch_size, self.n_in))

        self.output_q = self.activation_fn(
                T.dot(self.inpt_q, self.w) + self.b)

        self.inpt_t1 = inpts_t1.reshape((mini_batch_size * GOOD_TITLE_NUM, self.n_in))
        self.output_t1 = self.activation_fn(
                T.dot(self.inpt_t1, self.w) + self.b)

        self.inpt_t2 = inpts_t2.reshape((mini_batch_size * BAD_TITLE_NUM, self.n_in))
        self.output_t2 = self.activation_fn(
                T.dot(self.inpt_t2, self.w) + self.b)
    
    def cost(self, mini_batch_size):
        length_q = T.sqrt(T.sum(T.sqr(self.output_q), axis = 1) + EPSILON)
        length_t1 = T.sqrt(T.sum(T.sqr(self.output_t1), axis = 1) + EPSILON)
        length_t2 = T.sqrt(T.sum(T.sqr(self.output_t2), axis = 1) + EPSILON)
        cos_dist = theano.shared(0)
        for index in xrange(0, mini_batch_size):
            t1_slice = self.output_t1[index * GOOD_TITLE_NUM:(index + 1):GOOD_TITLE_NUM]
            t2_slice = self.output_t2[index * BAD_TITLE_NUM:(index + 1):BAD_TITLE_NUM]
            q_slice = self.output_q[index]

            length_t1_slice = length_t1[index * GOOD_TITLE_NUM:(index + 1)*GOOD_TITLE_NUM]
            length_t2_slice = length_t2[index * BAD_TITLE_NUM:(index + 1)*BAD_TITLE_NUM]
            length_q_slice = length_q[index]

            cos_dist -= T.sum((T.dot(t1_slice, q_slice) / (length_t1_slice * length_q_slice)))
            # cos_dist += T.sum((T.dot(t2_slice, q_slice) / (length_t2_slice * length_q_slice)))
            # cos_dist -= T.sum(length_t1_slice * length_q_slice)
            # cos_dist += T.sum(length_t2_slice * length_q_slice)
            cos_dist /= (BAD_TITLE_NUM * GOOD_TITLE_NUM)
        return cos_dist

    def accuracy(self):
        "Return the accuracy for the mini-batch."
        length_q = T.sqrt(T.sum(T.sqr(self.output_q), axis = 1) + EPSILON)
        length_t1 = T.sqrt(T.sum(T.sqr(self.output_t1), axis = 1) + EPSILON)
        length_t2 = T.sqrt(T.sum(T.sqr(self.output_t2), axis = 1) + EPSILON)
        cos_dist = theano.shared(0)
        for index in xrange(0, 10):
            t1_slice = self.output_t1[index * GOOD_TITLE_NUM:(index + 1):GOOD_TITLE_NUM]
            t2_slice = self.output_t2[index * BAD_TITLE_NUM:(index + 1):BAD_TITLE_NUM]
            q_slice = self.output_q[index]

            length_t1_slice = length_t1[index * GOOD_TITLE_NUM:(index + 1)*GOOD_TITLE_NUM]
            length_t2_slice = length_t2[index * BAD_TITLE_NUM:(index + 1)*BAD_TITLE_NUM]
            length_q_slice = length_q[index]

            cos_dist -= T.sum(BAD_TITLE_NUM * T.dot(t1_slice, q_slice) / (length_t1_slice * length_q_slice))
            cos_dist += T.sum(GOOD_TITLE_NUM * T.dot(t2_slice, q_slice) / (length_t2_slice * length_q_slice))
            cos_dist /= (BAD_TITLE_NUM * GOOD_TITLE_NUM)
        return cos_dist

class ClickData(object):

    def __init__(self, from_raw_file=True, query_feature_file_name="../data/query_feature", doc_feature_file_name="../data/doc_feature"):
        if from_raw_file:
            print "Get feature number."
            self.feature_num = self.find_global_max_feature_id(query_feature_file_name, doc_feature_file_name) + 1
            print "Feature number is %d" % (self.feature_num)
            print "Get query feature."
            self.query_feature_list = self.load_feature_file(query_feature_file_name, 0, 1000000)
            query_feature_file = open("../data/query_feature.cPickle", "wb")
            cPickle.dump(self.query_feature_list, query_feature_file)
            query_feature_file.close()
            print "Get doc feature."
            self.doc_feature_list = self.load_feature_file(doc_feature_file_name, 0, 100000000)
            doc_feature_file = open("../data/doc_feature.cPickle", "wb")
            cPickle.dump(self.doc_feature_list, doc_feature_file)
            doc_feature_file.close()
            print "Get maximum query id."
            self.max_query_id = self.find_max_sample_id(query_feature_file_name)
            print "Maximum query id is %d" % (self.max_query_id)
        else:
            print "Get global maximum feature id."
            self.feature_num = self.find_global_max_feature_id(query_feature_file_name, doc_feature_file_name)
            print "Global maximum feature id is %d" % (self.feature_num)
            print "Get query feature."
            query_feature_file = open("../data/query_feature.cPickle", "rb")
            self.query_feature_list = cPickle.load(query_feature_file)
            query_feature_file.close()
            print "Get doc feature."
            doc_feature_file = open("../data/doc_feature.cPickle", "rb")
            self.doc_feature_list = cPickle.load(doc_feature_file)
            doc_feature_file.close()

    def find_max_feature_id(self, feature_file_name):
        max_feature_id = 0
        for line in open(feature_file_name):
            token = line.split("\t")[-1]
            feature_id = int(token.split(":")[0])
            if feature_id > max_feature_id:
                max_feature_id = feature_id
        return max_feature_id

    def find_global_max_feature_id(self, query_feature_file_name="../data/query_feature", doc_feature_file_name="../data/doc_feature"):
        return max(self.find_max_feature_id(query_feature_file_name), self.find_max_feature_id(doc_feature_file_name))

    def find_max_sample_id(self, feature_file_name):
        return len(open(feature_file_name).readlines())

    def load_feature_file(self, feature_file_name, begin_line, end_line):
        feature_list = []
        for line in open(feature_file_name).readlines()[begin_line:end_line]:
            feature = dict()
            tokens = line.split("\t")
            for token in tokens:
                feature_pair = token.split(":")
                feature[int(feature_pair[0])] = int(feature_pair[1])
            feature_list.append(feature)
        return feature_list

    def transfer_feature_dict2list(self, feature_dict):
        feature_list = [0] * self.feature_num
        for feature_id, feature_value in feature_dict.items():
            feature_list[feature_id] = feature_value
        return feature_list

    def load_click_feature_file(self, feature_file_name, begin_qid = 0, end_qid = 100):
        click_feature_dict = {}
        GOOD_TITLE_ID = 0
        BAD_TITLE_ID = 1
        last_query_id = -1
        qid = -1
        for line in open(feature_file_name):
            tokens = line.split("\t")
            query_id = int(tokens[0])
            if query_id != last_query_id:
                qid += 1
                last_query_id = query_id
            if qid < begin_qid:
                continue
            elif qid >= end_qid:
                break
            doc_id = int(tokens[1])
            impression = int(tokens[2])
            examination = int(tokens[3])
            click = int(tokens[4])
            if query_id not in click_feature_dict:
                click_feature_dict[query_id] = [[], []]
            if click > 0:
                click_feature_dict[query_id][GOOD_TITLE_ID].append(doc_id)
            else:
                click_feature_dict[query_id][BAD_TITLE_ID].append(doc_id)
        click_query_feature_list = []
        click_title_feature_list = []
        noclick_title_feature_list = []
        query_id_list = []
        click_title_id_list = []
        noclick_title_id_list = []
        for query_id in click_feature_dict:
            if len(click_feature_dict[query_id][GOOD_TITLE_ID]) < GOOD_TITLE_NUM \
                    or len(click_feature_dict[query_id][BAD_TITLE_ID]) < BAD_TITLE_NUM:
                continue
            click_query_feature_list.append(self.transfer_feature_dict2list(self.query_feature_list[query_id]))
            query_id_list.append(query_id)
            for doc_id in click_feature_dict[query_id][GOOD_TITLE_ID][:GOOD_TITLE_NUM]:
                click_title_feature_list.append(self.transfer_feature_dict2list(self.doc_feature_list[doc_id]))
                click_title_id_list.append(doc_id)
            for doc_id in click_feature_dict[query_id][BAD_TITLE_ID][:BAD_TITLE_NUM]:
                noclick_title_feature_list.append(self.transfer_feature_dict2list(self.doc_feature_list[doc_id]))
                noclick_title_id_list.append(doc_id)
        return click_query_feature_list, click_title_feature_list, noclick_title_feature_list,\
                query_id_list, click_title_id_list, noclick_title_id_list

    def load_data_shared(self, training_data_file_name="../data/train_data",\
            validation_data_file_name="../data/valid_data", test_data_file_name="../data/valid_data"):
        print "Get training data."
        training_click_query_feature_list, training_click_title_feature_list, training_noclick_title_feature_list, \
                training_query_id_list, training_click_title_id_list, training_noclick_title_id_list =\
                self.load_click_feature_file(training_data_file_name)
        validation_click_query_feature_list, validation_click_title_feature_list, validation_noclick_title_feature_list, \
                validation_query_id_list, validation_click_title_id_list, validation_noclick_title_id_list =\
                self.load_click_feature_file(validation_data_file_name)
        test_click_query_feature_list, test_click_title_feature_list, test_noclick_title_feature_list, \
                test_query_id_list, test_click_title_id_list, test_noclick_title_id_list =\
                self.load_click_feature_file(test_data_file_name)
        def shared(data):
            """Place the data into shared variables.  This allows Theano to copy
                the data to the GPU, if one is available.
            """
            shared_data = theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
            return shared_data
        return [shared(training_click_query_feature_list), shared(training_click_title_feature_list), shared(training_noclick_title_feature_list),\
                shared(validation_click_query_feature_list), shared(validation_click_title_feature_list), shared(validation_noclick_title_feature_list),\
                shared(test_click_query_feature_list), shared(test_click_title_feature_list), shared(test_noclick_title_feature_list)]

class NetWork(object):

    def load_data(self):
        srng = RandomStreams(seed=234)
        rv_q = srng.uniform((10 * 10, 15000)).eval()
        rv_t1 = srng.normal((10 * 100, 15000)).eval()
        rv_t2 = srng.normal((10 * 40, 15000)).eval()
        
        shared_q = theano.shared(
            np.asarray(rv_q,
                       dtype=theano.config.floatX),
            name='q', borrow=True)

        shared_t1 = theano.shared(
            np.asarray(rv_t1,
                       dtype=theano.config.floatX),
            name='t1', borrow=True)

        shared_t2 = theano.shared(
            np.asarray(rv_t2,
                       dtype=theano.config.floatX),
            name='t2', borrow=True)

        return shared_q, shared_t1, shared_t2

    def __init__(self, click_data, layers, mini_batch_size, eta):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        
        self.q = T.matrix("q")
        self.t1 = T.matrix("t1")
        self.t2 = T.matrix("t2")

        init_layer = self.layers[0]
        
        init_layer.set_inpt(self.q, self.t1, self.t2, self.mini_batch_size)

        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output_q, prev_layer.output_t1, prev_layer.output_t2, self.mini_batch_size)

        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])

        shared_q, shared_t1, shared_t2 = self.load_data()
        # shared_q, shared_t1, shared_t2, shared_v_q, shared_v_t1, shared_v_t2, shared_t_q, shared_t_t1, shared_t_t2 = click_data.load_data_shared()
        print T.shape(shared_q).eval()
        print T.shape(shared_t1).eval()
        print T.shape(shared_t2).eval()

        num_training_batches = size(shared_q)/mini_batch_size
        print "num_training_batches: %d" % (num_training_batches)
        # num_training_batches = 10

        cost = layers[-1].cost(mini_batch_size) + 0.5 * l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)

        updates = [(param, param - eta*grad)
                        for param, grad in zip(self.params, grads)]

        i = T.lscalar()
        train = theano.function([i], cost, updates = updates,
                            givens = { self.q: shared_q[i * mini_batch_size: (i+1) * mini_batch_size], self.t1: shared_t1[i * mini_batch_size * GOOD_TITLE_NUM: (i+1)*mini_batch_size*GOOD_TITLE_NUM],\
                            self.t2: shared_t2[i * mini_batch_size * BAD_TITLE_NUM: (i+1)*mini_batch_size*BAD_TITLE_NUM]})

        for mini_batch_index in xrange(num_training_batches):
            cost = train(mini_batch_index)
            print cost

# click_data = ClickData(False)
click_data = None

# FIRST_LAY = FullyConnectedLayer(click_data.feature_num, 3000)
FIRST_LAY = FullyConnectedLayer(15000, 3000)
SECOND_LAY = FullyConnectedLayer(3000, 128)

net = NetWork(click_data, [FIRST_LAY, SECOND_LAY], 1, 0.1)
