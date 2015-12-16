import theano
import theano.tensor as T
from theano import printing
import numpy as np

EPSION = 0.000001

x = T.tensor3("x")

test_data = [[[1,2],[2,3],[3,4]], [[2,1],[2,2],[3,3]]]
shared_x = theano.shared(np.asarray(test_data, dtype=theano.config.floatX), borrow = True)

test_q = np.array([[1,2],[2,3]])

hello_world_op = printing.Print('hello world')
printed_x = hello_world_op(x)
print_f = theano.function([x], printed_x)

shared_xx = shared_x.reshape((6,2))
shared_q = theano.shared(np.asarray(test_q, dtype=theano.config.floatX), borrow = True)


fx = T.dmatrix("f")

length_fx = T.sqrt(T.sum(T.sqr(fx), axis=1) + EPSION)

length_function = theano.function([fx], length_fx)

length_q = length_function(shared_q.eval())
length_x = length_function(shared_xx.eval())

print length_q
print length_x

cx = T.dmatrix("cx")
cq = T.dvector("cq")
lq = T.dscalar("lq")
lx = T.dvector("lx")

cosine_dist = T.sum(T.dot(cx, cq) / (lx * lq))

cosine_function = theano.function([cx, cq, lq, lx], cosine_dist)

print cosine_function(shared_xx.eval()[0:3], shared_q.eval()[0], length_q[0], length_x[0:3])


