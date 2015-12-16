import theano
import theano.tensor as T
from theano import printing
import numpy as np

EPSION = 0.000001

x = T.tensor3("x")

test_data = [[[1,2],[2,3],[3,4]], [[2,1],[2,2],[3,3]]]
shared_x = theano.shared(np.asarray(test_data, dtype=theano.config.floatX), borrow = True)

inpt = T.matrix("inpt")

cost = inpt * inpt

i = T.lscalar("i")
fn = theano.function([i], cost, givens = {inpt:shared_x[i]})
z = fn(0)
#print z

test_q = np.array([[1,2],[2,3]])

hello_world_op = printing.Print('hello world')
printed_x = hello_world_op(x)
f = theano.function([x], printed_x)



shared_xx = shared_x.reshape((6,2))
shared_q = theano.shared(np.asarray(test_q, dtype=theano.config.floatX), borrow = True)

x = T.lmatrix("x")
print x.shape.eval({x: test_q})
print theano.function(inputs=[x], outputs=x.shape)(test_q)
print test_q.shape


fx = T.dmatrix("f")
print theano.function(inputs=[fx], outputs=fx.shape)(shared_xx.eval())


a1 = T.dvector("a1")
a2 = T.dvector("a2")

cosine_dist = T.dot(a1, a2) / (T.sqrt(T.dot(a1, a1)) * T.sqrt(T.dot(a2, a2)))

length_fx = T.sqrt(T.sum(T.sqr(fx), axis=1) + EPSION)

dot_function = theano.function([a1, a2], cosine_dist)

print dot_function(shared_q.eval()[0], shared_x.eval()[0][0])


length_function = theano.function([fx], length_fx)

print length_function(shared_q.eval())



