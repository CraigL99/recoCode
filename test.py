import numpy as np
import theano.tensor as T
import theano
import theano
import scipy.sparse as sp

m1 = T.dmatrix('m1')
m2 = T.dmatrix('m2')

z = m1 - m2
f = theano.function([m1, m2], z)

a = ([[1,2],
	[3,4]])


b = ([[1,1],
	[1,1]])

print f(a, b)

#Now try it with a sparse matrix
data2 = sp.lil_matrix((2,2))
data2[0,0] = 7
data2 = data2.todense()
print f(a, data2)

