import theano
import theano.tensor as T
a = T.vector()
b = T.vector()
out = a ** 2 + b**2 + 2*a*b
f = theano.function([a, b], out)
print f([0, 1, 2], [2, 3, 4])