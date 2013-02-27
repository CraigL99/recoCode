import theano
import theano.tensor as T
import numpy as np

x = T.matrix('x')
y = T.matrix('y')

z = x + y

f = theano.function([x,y],z)

a = [[1, 2], 
	[3,4]]
b = [[10,20], [30,40]]

print f(a,b)


n_users = 10
n_items = 20
n_fac = 10
U = theano.shared(value=np.zeros((
            n_users, n_fac), dtype=theano.config.floatX), name='U')

V = theano.shared(value=np.zeros((
            n_fac, n_items), dtype=theano.config.floatX), name='V')

print "\nU:"
print U.get_value()

print "\nV:"
print V.get_value()


data = np.random.randn(n_users, n_items).astype(theano.config.floatX)
print data

U.set_value(0.01 * np.random.randn(n_users, n_fac))
V.set_value(0.01 * np.random.randn(n_fac, n_items))

print "new U:"
print U.get_value()


print "new V:"
print V.get_value()

#Now how on earth does the actual factoring happen?


