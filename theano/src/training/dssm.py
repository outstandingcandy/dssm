import sys
sys.path.append("/home/search/tangjie/dssm/src/")
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
from util.util import *
from theano import pp

sys.setrecursionlimit(10000)

EPSILON = 0.0000000001

theano.config.floatX = 'float32'
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'
# theano.config.mode = 'DebugMode'
# theano.config.on_unused_input = 'warn'
# theano.config.profile  = 'True'
theano.config.openmp_elemwise_minsize = '64'

def size(data):
    "Return the size of the dataset `data`."
    # return TS.csm_shape(data)[0].eval()
    return data.shape[0].eval()

class SparseFullyConnectedLayer(object):
    
    def __init__(self, n_in, n_out, input_data_list, activation_fn=tanh):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.w = theano.shared(
            np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6.0/(n_in+n_out)), high=np.sqrt(6.0/(n_in+n_out)), size=(n_in, n_out)),
                    dtype=theano.config.floatX),
            name='w', borrow=True)
        # self.b = theano.shared(
        #     np.asarray(np.zeros((n_out)), dtype=theano.config.floatX),
        #     name='b', borrow=True)
        # self.params = [self.w, self.b]
        # self.params = [self.w]

        self.b = theano.shared(
                    np.asarray(np.random.normal(loc=0.0, scale=1.0/(n_in+n_out), size=(n_out,)),
                    dtype=theano.config.floatX),
                    name='b', borrow=True)
        self.params = [self.w, self.b]

        # self.w = T._shared(
        #     np.asarray(
        #         np.random.uniform(
        #             low=-np.sqrt(6.0/(n_in+n_out)), high=np.sqrt(6.0/(n_in+n_out)), size=(n_in, n_out)),
        #             dtype=theano.config.floatX),
        #     name='w', borrow=True)
        # self.b = T._shared(
        #     np.asarray(np.zeros((n_out)), dtype=theano.config.floatX),
        #     name='b', borrow=True)

        self.q, self.d = input_data_list
        self.output = [self.activation_fn(T.add(TS.basic.structured_dot(self.q, self.w), self.b)), \
                self.activation_fn(T.add(TS.basic.structured_dot(self.d, self.w), self.b))]

    def watch_input(self):
        return self.q, self.d

class CosineLayer(object):

    def __init__(self, input_data_list, mini_batch_size, negative_d_num):
        self.q, self.d = input_data_list
        self.mini_batch_size = mini_batch_size 
        self.negative_d_num = negative_d_num

    """
    Given two inputs [q1 q2 .... qmb] and [d1 d2 ... dmb],
    this class computes the pairwise cosine similarity 
    between positive pairs and neg pairs
    """
    def compute_cosine_between_two_vectors(self, q_ind, d_ind, Q, D):
        q = Q[q_ind]
        d = D[d_ind]
        qd_dot = T.dot(q, d)
        q_norm = T.sqrt((q**2).sum() + EPSILON)
        d_norm = T.sqrt((d**2).sum() + EPSILON)
        return qd_dot / (q_norm * d_norm)

    def compute_cosine_between_matrixes(self, q_matrix, d_matrix):
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
        column1 = T.log(T.exp(components_reshape)[:,0] / T.sum(T.exp(components_reshape), axis=1))
        # get the final output
        self.output_train = - column1.sum()
        return self.output_train / self.mini_batch_size

    def cost_v2(self):
        q_matrix = self.q
        d_matrix = self.d[0::self.negative_d_num + 1]
        consine_vector = self.compute_cosine_between_matrixes(q_matrix, d_matrix)
        for i in range(1, self.negative_d_num + 1):
            q_matrix = self.q
            d_matrix = self.d[i::self.negative_d_num + 1]
            consine_vector = T.concatenate([consine_vector, self.compute_cosine_between_matrixes(q_matrix, d_matrix)])
        components_reshape = T.reshape(consine_vector, (self.negative_d_num+1, self.mini_batch_size)).T
        # components_reshape = T.reshape(consine_vector, (self.negative_d_num + 1, self.mini_batch_size))
        # for this matrix, each line is a prob distribution right now.
        components_reshape_softmax = T.nnet.softmax(components_reshape)
        # get the first column
        column1 = T.log(components_reshape_softmax[:,0])

        # column1 = T.log(T.exp(components_reshape[:, 0]) / T.sum(T.exp(components_reshape), axis=1))

        # get the final output
        self.output_train = -column1.sum()
        return self.output_train / self.mini_batch_size
        # return self.output_train

    def cost_v3(self):
        q_matrix = self.q
        d_matrix = self.d[0::self.negative_d_num + 1]
        consine_vector = self.compute_cosine_between_matrixes(q_matrix, d_matrix)
        p_sum = T.exp(consine_vector)
        n_sum_1 = self.compute_cosine_between_matrixes(self.q, self.d[1::self.negative_d_num+1])
        n_sum_2 = self.compute_cosine_between_matrixes(self.q, self.d[2::self.negative_d_num+1])
        n_sum_3 = self.compute_cosine_between_matrixes(self.q, self.d[3::self.negative_d_num+1])
        n_sum_4 = self.compute_cosine_between_matrixes(self.q, self.d[4::self.negative_d_num+1])
        n_sum = T.add(p_sum, T.exp(n_sum_1), T.exp(n_sum_2), T.exp(n_sum_3), T.exp(n_sum_4))
        column1 = T.log(p_sum / n_sum)
        self.output_train = -column1.sum()
        return self.output_train / self.mini_batch_size
        # return self.output_train

    def test(self):
        return self.compute_cosine_between_two_vectors(0, 0, self.q, self.d)

    def test_batch(self):
        return self.compute_cosine_between_matrixes(self.q, self.d)

    def test_v2(self):
        q_matrix = self.q
        d_matrix = self.d[0::self.negative_d_num + 1]
        consine_vector = self.compute_cosine_between_matrixes(q_matrix, d_matrix)
        for i in range(1, self.negative_d_num + 1):
            q_matrix = self.q
            d_matrix = self.d[i::self.negative_d_num + 1]
            consine_vector = T.concatenate([consine_vector, self.compute_cosine_between_matrixes(q_matrix, d_matrix)])
        components_reshape = T.reshape(consine_vector, (self.negative_d_num + 1, self.mini_batch_size)).T
        gt_1 = T.sum(T.gt(components_reshape[:, 0], components_reshape[:, 1]))
        gt_2 = T.sum(T.gt(components_reshape[:, 0], components_reshape[:, 2]))
        gt_3 = T.sum(T.gt(components_reshape[:, 0], components_reshape[:, 3]))
        gt_4 = T.sum(T.gt(components_reshape[:, 0], components_reshape[:, 4]))
        gt_sum = gt_1 + gt_2 + gt_3 + gt_4
        return components_reshape, gt_sum * 1.0 / (self.mini_batch_size * self.negative_d_num)

    def test_v3(self):
        q_matrix = self.q
        d_matrix = self.d[0::self.negative_d_num + 1]
        consine_vector = self.compute_cosine_between_matrixes(q_matrix, d_matrix)
        p_sum = consine_vector
        n_sum_1 = self.compute_cosine_between_matrixes(self.q, self.d[1::self.negative_d_num+1])
        n_sum_2 = self.compute_cosine_between_matrixes(self.q, self.d[2::self.negative_d_num+1])
        n_sum_3 = self.compute_cosine_between_matrixes(self.q, self.d[3::self.negative_d_num+1])
        n_sum_4 = self.compute_cosine_between_matrixes(self.q, self.d[4::self.negative_d_num+1])
        a_sum = T.reshape(T.concatenate([p_sum, n_sum_1, n_sum_2, n_sum_3, n_sum_4]), (self.negative_d_num + 1, self.mini_batch_size)).T
        gt_1 = T.sum(T.gt(p_sum, n_sum_1))
        gt_2 = T.sum(T.gt(p_sum, n_sum_2))
        gt_3 = T.sum(T.gt(p_sum, n_sum_3))
        gt_4 = T.sum(T.gt(p_sum, n_sum_4))
        gt_sum = gt_1 + gt_2 + gt_3 + gt_4
        return a_sum, gt_sum * 1.0 / (self.mini_batch_size * self.negative_d_num)

class FullyConnectedLayer(object):
    
    def __init__(self, n_in, n_out, input_data_list, activation_fn=tanh):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6.0/(n_in+n_out)), high=np.sqrt(6.0/(n_in+n_out)), size=(n_in, n_out)),
                    dtype=theano.config.floatX),
            name='w', borrow=True)
        # self.b = theano.shared(
        #     np.asarray(np.zeros((n_out)), dtype=theano.config.floatX),
        #     name='b', borrow=True)
        # self.params = [self.w, self.b]
        # self.params = [self.w]

        self.b = theano.shared(
                    np.asarray(np.random.normal(loc=0.0, scale=1.0/(n_in+n_out), size=(n_out,)),
                    dtype=theano.config.floatX),
                    name='b', borrow=True)
        self.params = [self.w, self.b]

        self.q, self.d = input_data_list
        self.output = [self.activation_fn(T.dot(self.q, self.w) + self.b), \
                self.activation_fn(T.dot(self.d, self.w) + self.b)]

    def watch_input(self):
        return self.q, self.d
    
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

    def __init__(self, q, d, feature_num, negative_d_num, mini_batch_size, hidden_layer_output_num=[300, 300, 128]):
        self.hidden_layers = []
        self.q = q
        self.d = d
        layer_input_num = feature_num
        layer_output_num = hidden_layer_output_num[0]
        self.hidden_layers.append(SparseFullyConnectedLayer( \
                layer_input_num, layer_output_num, [self.q, self.d], tanh))
        # self.hidden_layers.append(FullyConnectedLayer( \
        #         layer_input_num, layer_output_num, [TS.dense_from_sparse(self.q), TS.dense_from_sparse(self.d)]))
        # self.hidden_layers.append(FullyConnectedLayer( \
        #         layer_input_num, layer_output_num, [self.q, self.d]))

        layer_input = self.hidden_layers[-1].output
        layer_input_num = layer_output_num
        for layer_output_num in hidden_layer_output_num[1:]:
            hidden_layer = FullyConnectedLayer(layer_input_num, \
                    layer_output_num, layer_input, tanh)
            self.hidden_layers.append(hidden_layer)
            layer_input = self.hidden_layers[-1].output
            layer_input_num = layer_output_num
        self.cosine_layer = CosineLayer(self.hidden_layers[-1].output, mini_batch_size, negative_d_num)
        self.params = []
        for hl in self.hidden_layers:
            self.params.extend(hl.params)

    def set_input(self, q, d):
        self.q = q
        self.d = d

def train(training_data_file_name="../data/combined_feature_1m", model_name="alpha", mini_batch_size=4096, eta=0.1, sample_size=64*1024):
    print_log("Initialize training data")
    training_data_set =\
            training_data.TrainingData(\
                    combined_feature_file_name=training_data_file_name,\
                    sample_size=sample_size)
    negative_d_num = 4
    train_q = theano.shared(sp.csr_matrix((sample_size, training_data_set.feature_num), dtype=theano.config.floatX), borrow=True)
    train_d = theano.shared(sp.csr_matrix(((negative_d_num+1)*sample_size, training_data_set.feature_num), dtype=theano.config.floatX), borrow=True)
    # train_q = theano.shared(np.asarray(np.empty((sample_size, training_data_set.feature_num)), dtype=theano.config.floatX), borrow=True)
    # train_d = theano.shared(np.asarray(np.empty(((negative_d_num+1)*sample_size, training_data_set.feature_num)), dtype=theano.config.floatX), borrow=True)

    q = TS.csr_matrix("q")
    d = TS.csr_matrix("d")
    # q = T.matrix("q")
    # d = T.matrix("d")

    print_log("Create training function")
    net = NetWork(q, d, training_data_set.feature_num, negative_d_num, mini_batch_size, hidden_layer_output_num=[300, 300, 128])

    # indexes = generate_index(mini_batch_size, negative_d_num)
    # cost = net.cosine_layer.cost(indexes[0], indexes[1])
    # cost = net.cosine_layer.cost_v2()
    cost = net.cosine_layer.cost_v3()
    test = net.cosine_layer.test_v3()
    watch_input = net.hidden_layers[0].watch_input()
    grads = theano.grad(cost, net.params)

    # print_log("%s" % pp(grads[0]))
    # f = theano.function([q], grads[0])
    # pp(f.maker.fgraph.outputs[0])
    # theano.printing.pydotprint(grads[0], outfile="symbolic_graph_unopt.png", var_with_name_simple=True) 

    updates = [(param, param-eta*grad)
                    for param, grad in zip(net.params, grads)]
    # updates = [(param, param-1*eta*grad)
    #                 for param, grad in zip(net.params[:1], grads[:1])]
    # updates += [(param, param-eta*grad)
    #                 for param, grad in zip(net.params[1:], grads[1:])]
    i = T.lscalar()
    train = theano.function([i], cost, updates = updates,\
                        givens = {q:train_q[i*mini_batch_size:(i+1)*mini_batch_size],\
                        d:train_d[i*mini_batch_size*(negative_d_num+1):(i+1)*mini_batch_size*(negative_d_num+1)]})
    test = theano.function([i], test,\
                        givens = {q:train_q[i*mini_batch_size:(i+1)*mini_batch_size],\
                        d:train_d[i*mini_batch_size*(negative_d_num+1):(i+1)*mini_batch_size*(negative_d_num+1)]})
    watch_input = theano.function([i], watch_input,\
                        givens = {q:train_q[i*mini_batch_size:(i+1)*mini_batch_size],\
                        d:train_d[i*mini_batch_size*(negative_d_num+1):(i+1)*mini_batch_size*(negative_d_num+1)]})
    iteration = 0
    epochs = 1000
    for epoch in xrange(epochs):
        training_data_set.load_sparse_training_data_v4(train_q, train_d)
        # print_log("%s" % train_q.eval()[11])
        # print_log("%s" % train_d.eval()[55])
        # print_log("%s" % train_d.eval()[59])
        while size(train_q) > 0:
            iteration += 1
            num_training_batches = size(train_q)/mini_batch_size
            print_log("epoch: %d\titeration: %d" % (epoch, iteration))
            print_log("num_training_batches: %d" % (num_training_batches))
            print_log("eta: %f" % (eta))
            # for i in range(0, len(net.params), 2):
            #     print_log("layer %d w params:\n%s" % (i/2 + 1, np.asarray(net.params[i].eval())))
            #     print_log("layer %d b params:\n%s" % (i/2 + 1, np.asarray(net.params[i+1][:8].eval())))
            # for i in range(0, len(net.params), 1):
            #     print_log("layer %d w params:\n%s" % (i + 1, np.asarray(net.params[i].eval())))
            loss = 0
            for mini_batch_index in xrange(num_training_batches):
                if mini_batch_index == 0:
                    output_test = test(0)
                    print_log("test output:\n%s" % output_test[0])
                    print_log("validate:\t%f" % output_test[1])
                    continue

                # w_q, w_d = watch_input(mini_batch_index)
                # print_log("q\n%s" % w_q[11])
                # print_log("pd\n%s" % w_d[55])
                # print_log("nd\n%s" % w_d[59])

                loss += train(mini_batch_index)

            print_log("loss:\t%f" % (loss / num_training_batches))
            training_data_set.load_sparse_training_data_v4(train_q, train_d)

        model_file = open("../../model/dssm_%s_%d_%f_%d_%d.save" % (model_name, mini_batch_size, eta, epoch, iteration), "wb")
        cPickle.dump(net, model_file)
        model_file.close()
        training_data_set.clear()

if __name__ == "__main__":
    model_name = sys.argv[1]
    mini_batch_size = int(sys.argv[2])
    eta = float(sys.argv[3])
    train(training_data_file_name="../../data/combined_feature_50m_newhash", model_name=model_name, mini_batch_size=mini_batch_size, sample_size=64*1024, eta=eta)
