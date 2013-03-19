import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from time import sleep
import scipy.sparse as sp
from theano import sparse

from matrixfacSparse_addValidateTest import MatrixFactorization, readData


class HiddenLayer(object):

	def __init__():
		pass
	pass

class MLP(object):

	def __init__(self, n_users, n_items, n_fac):
		#self.hiddenLayer = HiddenLayer()
		
		self.modelLayer = MatrixFactorization(n_users=n_users, n_items=n_items, n_fac=n_fac)


def main():
	learning_rate = 0.0001
	n_users = 943
	n_items = 1682
	n_fac = 30
	batch_size = 10000 #must be a multiple of the total number of ratings

	data = readData('u.data', n_users, n_items)
	s_d = data[0] 
	r_d = data[1]
	c_d = data[2]
	actualMatrix = data[3]

	data2 = readData('validate.data', n_users, n_items)
	s_d_validate = data2[0] 
	r_d_validate = data2[1]
	c_d_validate = data2[2]

	data3 = readData('test.data', n_users, n_items)
	s_d_test = data3[0] 
	r_d_test = data3[1]
	c_d_test = data3[2]

	#compute number of minibatches for traiing, validaiton and testing
	n_train_batches = s_d.get_value().shape[0] / batch_size
	n_test_batches = s_d_test.get_value().shape[0] / batch_size
	n_valid_batches = s_d_validate.get_value().shape[0] / batch_size 

	print "Number of ratings: " + str(s_d.get_value().shape[0])
	print "Batch size: " + str(batch_size)
	print "n_train_batchs: " + str(n_train_batches)
	print "n_test_batchs: " + str(n_test_batches)
	print "n_valid_batchs: " + str(n_valid_batches)

	###################
	#Build actual model
	###################
	
	#allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch
	sparseData = T.vector('sparseData', dtype=theano.config.floatX)
	rowInd = T.vector('rowInd', dtype='int32')  # symbolic var for row indices
	colInd = T.vector('colInd', dtype='int32')  # symbolic var for col indices

	model = MLP(n_users=n_users, n_items=n_items, n_fac=n_fac)
	
if __name__ == '__main__':
	main()

