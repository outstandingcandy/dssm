import theano
import theano.tensor as T
from theano import printing
import numpy as np

x = T.lscalar("x")
y = T.lscalar("y")
z = theano.shared(0)
k = T.lscalar("k")
fun1 = y ** 2
fun = 2 * x + fun1
fx = theano.function([k,y], fun, updates = [(z, x + y)], givens = [(x, k)])
a = fx(1,2)
print a

v = T.ivector("v")

data = [1, 2, ,3 ,4]
dx = theano.shared(np)


