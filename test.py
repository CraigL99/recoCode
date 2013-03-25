import theano.tensor as T
import numpy as np
from theano import function
import theano

"""
print "testing elementwise multiplication"
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x * y
f = function([x, y], z)

print f(np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]]))

print "\ntesting dot"
U = theano.shared(value=np.zeros((
            2, 3), dtype=theano.config.floatX), name='U')

U.set_value(0.01 * np.random.randn(2, 3))


print U.get_value()
IBU = np.zeros((2,2), dtype=int)
IBU[0,0] = 1
print IBU


UD = T.dot(x, y)
f2 = function([x,y], UD)

print f2(IBU, U.get_value())

def testMatrixMult():
	x = T.dmatrix('x')
 	y = T.dmatrix('y')
	z = x * y
	f = function([x, y], z)
	print "testing matrix multiplication"
	print f(np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]]))

def testSumAlongRows():
	x = T.dmatrix('x')
	z = T.sum(x, axis=1)
	f = function([x,], z)
	m = np.asarray([[1, 2], [3, 4]])

	print "testing matrix sum along rows"
	print f(m)


"""

U = theano.shared(value=np.zeros((
            2, 2), dtype=theano.config.floatX), name='U')
U.set_value(0.01 * np.random.randn(2, 2))

V = theano.shared(value=np.zeros((
            2, 2), dtype=theano.config.floatX), name='U')
V.set_value(0.01 * np.random.randn(2, 2))

def squared_error(sparseData, IBM, UBM):
	UD = T.dot(UBM, U)
	VD = T.dot(V, IBM)
	#return UD
	matrix = UD*VD

	pred = T.sum(matrix, axis=1)
	return T.sum(T.square(sparseData-pred))

def main():
	s_d = np.asarray([1.0, 1.0])
	#s_d = theano.shared( np.asarray(s_d, dtype=theano.config.floatX) )
	
	IBM = np.zeros((2,2), dtype=theano.config.floatX) #item bit matrix
	IBM[0,0] = 1
	UBM = np.zeros((2,2), dtype=theano.config.floatX)
	UBM[0,0] = 1 #user bit matrix

	sparseData = T.vector('sparseData', dtype=theano.config.floatX)
	itemBitMatrix = T.matrix('itemBitMatrix', dtype=theano.config.floatX)
	userBitMatrix = T.matrix('itemBitMatrix', dtype=theano.config.floatX)

	cost = squared_error(sparseData, itemBitMatrix, userBitMatrix)
	index = T.lscalar()
	train_model = function (inputs=[index], outputs=cost, givens={ sparseData: s_d*index, itemBitMatrix: IBM*index, userBitMatrix: UBM*index}, on_unused_input='warn')
	print "and the output is..."
	print train_model(1)

if __name__ =='__main__':
	main()