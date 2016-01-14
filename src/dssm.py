import sys
import theano
import numpy as np
import scipy.sparse as sp
from theano import tensor as T
from theano import sparse as TS
from theano.tensor import tanh
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle
import training_data
from util import *

sys.setrecursionlimit(10000)

GOOD_TITLE_NUM = 1
BAD_TITLE_NUM = 1
EPSILON = 0.0000000001

theano.config.floatX = 'float32'
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
# theano.config.mode = 'DebugMode'
theano.config.on_unused_input = 'warn'
# theano.config.profile  = 'True'
theano.config.openmp_elemwise_minsize = '512'

def size(data):
    "Return the size of the dataset `data`."
    return TS.csm_shape(data)[0].eval()

class SparseFullyConnectedLayer(object):
    
    def __init__(self, n_in, n_out, input_data_list, activation_fn=tanh):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype="float32"),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype="float32"),
            name='b', borrow=True)
        self.params = [self.w, self.b]
        self.q, self.d = input_data_list
        self.output = [self.activation_fn(T.add(TS.basic.structured_dot(self.q, self.w), self.b)), \
                self.activation_fn(T.add(TS.basic.structured_dot(self.d, self.w), self.b))]

class CosineLayer(object):

    def __init__(self, input_data_list, mini_batch_size, negative_d_num, indexes):
        self.q, self.d = input_data_list
        self.mini_batch_size = mini_batch_size 
        self.negative_d_num = negative_d_num
        self.indexes = indexes
        self.cost(indexes[0], indexes[1])
        self.test(indexes[2], indexes[3])

    """
    Given two inputs [q1 q2 .... qmb] and [d1 d2 ... dmb],
    this class computes the pairwise cosine similarity 
    between positive pairs and neg pairs
    """
    def compute_cosine_between_two_vectors(self, q_ind, d_ind, q_matrix, d_matrix):
        # index is like (1,2)
        q = q_matrix[q_ind]
        d = d_matrix[d_ind]
        qd_dot = T.dot(q,d)
        q_norm = T.sqrt((q**2).sum() + EPSILON)
        d_norm = T.sqrt((d**2).sum() + EPSILON)
        return qd_dot / (q_norm * d_norm)

    def compute_cosine_between_matrixes(self, q_matrix, d_matrix):
        # index is like (1,2)
        qd_dot_vector = T.sum(q_matrix * d_matrix, 1)
        q_norm_vector = T.sqrt(T.sum(T.sqr(q_matrix), 1) + EPSILON)
        d_norm_vector = T.sqrt(T.sum(T.sqr(d_matrix), 1) + EPSILON)
        return qd_dot_vector / (q_norm_vector * d_norm_vector)

    # for train, we need to compute a cosine matrix for (Q,D), then compute a final score
    def cost(self, q_index_list, d_index_list):
        components, updates = theano.scan(self.compute_cosine_between_two_vectors,
                outputs_info=None,
                sequences=[q_index_list, d_index_list],
                non_sequences=[self.q, self.d])
        components_reshape = T.reshape(components, (self.mini_batch_size, self.negative_d_num+1))
        # for this matrix, each line is a prob distribution right now.
        # components_reshape_softmax = T.nnet.softmax(components_reshape)
        # get the first column
        # column1 = components_reshape_softmax[:,0]
        column1 = T.log(T.exp(components_reshape)[:,0] / T.sum(T.exp(components_reshape)))
        # get the final output
        self.output_train = - column1.sum()
        return self.output_train

    def cost_v2(self):
        q_matrix = self.q
        d_matrix = self.d[0::self.negative_d_num + 1]
        consine_vector = self.compute_cosine_between_matrixes(q_matrix, d_matrix)
        for i in range(1, self.negative_d_num + 1):
            q_matrix = self.q
            d_matrix = self.d[i::self.negative_d_num + 1]
            consine_vector = T.concatenate([consine_vector, self.compute_cosine_between_matrixes(q_matrix, d_matrix)])
        components_reshape = T.reshape(consine_vector, (self.negative_d_num + 1, self.mini_batch_size)).T
        # for this matrix, each line is a prob distribution right now.
        # components_reshape_softmax = T.nnet.softmax(components_reshape)
        # get the first column
        # column1 = components_reshape_softmax[:,0]
        column1 = T.log(T.exp(components_reshape)[:,0] / T.sum(T.exp(components_reshape)))
        # get the final output
        self.output_train = - column1.sum()
        return self.output_train

    # for test, we only need to compute a cosine vector for (Q,D)
    def test(self, q_index_list, d_index_list):
        # components is a vector         
        components, updates = theano.scan(self.compute_cosine_between_two_vectors,
                outputs_info=None,
                sequences=[q_index_list, d_index_list],
                non_sequences=[self.q, self.d])
        # get the final output
        self.output_test = components

    def test_v2(self):
        q_matrix = self.q
        d_matrix = self.d[0::self.negative_d_num + 1]
        consine_vector = self.compute_cosine_between_matrixes(q_matrix, d_matrix)
        for i in range(1, self.negative_d_num + 1):
            q_matrix = self.q
            d_matrix = self.d[i::self.negative_d_num + 1]
            consine_vector = T.concatenate([consine_vector, self.compute_cosine_between_matrixes(q_matrix, d_matrix)])
        components_reshape = T.reshape(consine_vector, (self.negative_d_num + 1, self.mini_batch_size)).T
        return components_reshape


class FullyConnectedLayer(object):
    
    def __init__(self, n_in, n_out, input_data_list, activation_fn=tanh):
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
        self.q, self.d = input_data_list
        self.output = [self.activation_fn(T.dot(self.q, self.w) + self.b), \
                self.activation_fn(T.dot(self.d, self.w) + self.b)]
    
def generate_index(mini_batch_size, negative_d_num):
    # Next, we need to generate 2 lists of index
    # these 2 lists together have mbsize*(neg+1) element
    # after reshape, it should be (mbsize, neg+1) matrix
    train_q_index = np.arange(0)
    train_d_index = np.arange(0)

    for tmp_index in range(mini_batch_size):
        # for current sample, it's positive pair is itself
        train_q_index = np.append(train_q_index, [tmp_index] * (negative_d_num + 1))
        train_d_index = np.append(train_d_index, [(tmp_index * (negative_d_num + 1) + y) for y in range(negative_d_num + 1)])
    test_q_index = np.arange(mini_batch_size)
    test_d_index = np.arange(mini_batch_size)
    print train_q_index, train_d_index

    indexes = [theano.shared(train_q_index), theano.shared(train_d_index), \
            theano.shared(test_q_index), theano.shared(test_d_index)]
    return indexes

class NetWork(object):

    def init_data(self, data_size, feature_size):
        data = []
        for i in range(data_size):
            data.append([0] * feature_size)
        return data

    def __init__(self, training_data, mini_batch_size=1024, eta=0.1, hidden_layer_output_num=[300, 300, 128], negative_d_num=4):
        self.mini_batch_size = mini_batch_size
        self.eta = eta
        self.negative_d_num = negative_d_num
        self.hidden_layers = []
        self.training_data = training_data
        self.q = TS.csr_matrix("q")
        self.d = TS.csr_matrix("d")
        self.input_data_list = [self.q, self.d]
        layer_input_num = self.training_data.feature_num
        layer_output_num = hidden_layer_output_num[0]
        self.hidden_layers.append(SparseFullyConnectedLayer( \
                layer_input_num, layer_output_num, self.input_data_list))

        layer_input = self.hidden_layers[-1].output
        layer_input_num = layer_output_num
        for layer_output_num in hidden_layer_output_num[1:]:
            hidden_layer = FullyConnectedLayer(layer_input_num, \
                    layer_output_num, layer_input)
            self.hidden_layers.append(hidden_layer)
            layer_input = hidden_layer.output
            layer_input_num = layer_output_num

        indexes = generate_index(mini_batch_size, negative_d_num)
        self.cosine_layer = CosineLayer(self.hidden_layers[-1].output, mini_batch_size, negative_d_num, indexes)

    def train(self):
        self.params = []
        for hl in self.hidden_layers:
            self.params.extend(hl.params)
        # cost = self.cosine_layer.cost(indexes[0], indexes[1])
        cost = self.cosine_layer.cost_v2()
        test = self.cosine_layer.test_v2()
        grads = T.grad(cost, self.params)
        updates = [(param, param - self.eta*grad)
                        for param, grad in zip(self.params, grads)]
        i = T.lscalar()
        print_log("Loading training data")
        q, d = self.training_data.load_sparse_training_data_v3(4096*16)
        print_log("Create training function")
        train = theano.function([i], cost, updates = updates,\
                            givens = { self.q: q[i * self.mini_batch_size: (i+1) * self.mini_batch_size], self.d: d[i * self.mini_batch_size * (self.negative_d_num + 1): (i+1)*self.mini_batch_size*(self.negative_d_num + 1)] })
        test = theano.function([i], test, \
                            givens = { self.q: q[i * self.mini_batch_size: (i+1) * self.mini_batch_size], self.d: d[i * self.mini_batch_size * (self.negative_d_num + 1): (i+1)*self.mini_batch_size*(self.negative_d_num + 1)] })
        iteration = 0
        epochs = 100
        for epoch in xrange(epochs):
            while size(q) > 0:
                iteration += 1
                num_training_batches = size(q)/self.mini_batch_size
                print_log("epoch: %d\titeration: %d" % (epoch, iteration))
                print_log("num_training_batches: %d" % (num_training_batches))
                loss = 0
                for mini_batch_index in xrange(num_training_batches):
                    loss += train(mini_batch_index) / num_training_batches
                print_log("loss:\t%f" % loss)
                output_test = test(0)
                print output_test
                q, d = training_data.load_sparse_training_data_v3(4096*16)
            break
            model_file = open("dssm_%d.save" % (epoch), "wb")
            cPickle.dump(self, model_file)
            model_file.close()
            training_data.clear()
            q, d = training_data.load_sparse_training_data_v3(4096*16)

if __name__ == "__main__":
    training_data = training_data.TrainingData()
    net = NetWork(training_data)
    net.train()
