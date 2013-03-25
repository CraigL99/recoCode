import numpy as np
import theano
import theano.tensor as T
import scipy.sparse as sp

class MatrixFactorization(object):

    def __init__(self, n_users, n_items, n_fac):

        self.U = theano.shared(value=np.zeros((
            n_users, n_fac), dtype=theano.config.floatX), name='U')

        self.V = theano.shared(value=np.zeros((
            n_fac, n_items), dtype=theano.config.floatX), name='V')

        self.pred = T.dot(self.U, self.V)

    def squared_error(self, sparseData, rowInd, colInd):
        return T.sum(T.square(sparseData-self.pred[rowInd, colInd]))

    def errors(self, sparseData, rowInd, colInd):
        return T.sum(T.square(sparseData-self.pred[rowInd, colInd]))

# Read movie Lens data.
# Pre, u.data rating file in same directory
# Post, returns 3 theano symbolic shared variables in a tuple, [data, row indices, column indices]
		#NOTE*** each row/column number has been subtracted by 1 s.t. they start at 0
def readData( filename, n_users, n_items ):
    f = open (filename, 'r') # user id | item id | rating | timestamp.
    sparseData = []
    rowInd = []
    colInd = []

    for line in f:
        lst = line.split("\t", 3)
        rowInd.append( int(lst[0])-1 )
        colInd.append( int(lst[1])-1 )
        sparseData.append( int(lst[2]) )

    #create the matrix used for visual display
    sparse_csc = sp.csc_matrix((sparseData, (rowInd, colInd)), shape=(n_users, n_items))
    actualMatrix = sparse_csc.todense()

    #convert the data into shared theano variables to enable use in cpu/gpu
    shared_sparseData = theano.shared( np.asarray(sparseData, dtype=theano.config.floatX) )
    shared_rowInd = theano.shared( np.asarray(rowInd, dtype=theano.config.floatX) )
    shared_colInd = theano.shared( np.asarray(colInd, dtype=theano.config.floatX) )

    return shared_sparseData, shared_rowInd, shared_colInd, actualMatrix

def main():
	#global config variables
	learning_rate = 0.0001
	n_users = 943
	n_items = 1682
	n_fac = 30
	batch_size = 10000 #must be a multiple of the total number of ratings

	#read in the data and convert it to bit vectors:
	data = readData(filename='u.data', n_users=n_users, n_items=n_items)
	s_d = data[0]
	r_d = data[1]
	c_d = data[2]

	#create a matrix to hold user bit vectors, size 0..99 999 x 0..1681
	userBitMatrix = np.zeros((s_d.get_value().shape[0], n_users), dtype=int)
	userBitMatrix[np.arange(s_d.get_value().shape[0]), (r_d.get_value()).astype(int)] = 1

	#create a matrix to hold user bit vectors, size 0..99 999 x 0..942
	itemBitMatrix = np.zeros((s_d.get_value().shape[0], n_items), dtype=int)
	itemBitMatrix[np.arange(s_d.get_value().shape[0]), (r_d.get_value()).astype(int)] = 1
	
	#cast rows to integers so they can be used as usual for indexing. (this used to be done within
		#the read method however in order to create the bitMatrix, had to do it afterwards)

	r_d = T.cast(r_d, 'int32') 
	c_d = T.cast(c_d, 'int32')

	#compute number of baches
	n_train_batches = s_d.get_value().shape[0] / batch_size


	print "Number of ratings: " + str(s_d.get_value().shape[0])
	print "Batch size: " + str(batch_size)
	print "n_train_batchs: " + str(n_train_batches)

	"""for each batch
			-convert the dataset to rowVectors, symbolicly
			-dot that with U/V, symbolicly
			-take the two new matricis and multiply them together elementwise,
			-sum the rows to create a enw vector, sybolicaly
			-compute the cost as the squared error as the actual vector - the created, squared
			-take the gradient of cost wrt u and v and then subtract that amount from the current u and v	
	"""
	#TODO need GT history here.

	"""model = MatrixFactorization(n_users=n_users, n_items=n_items, n_fac=n_fac)


    cost = model.squared_error(sparseData, rowInd, colInd) + L2 * lambda_L2 + L1 * lambda_L1
	g_U = T.grad(cost=cost, wrt=model.U)
    g_V = T.grad(cost=cost, wrt=model.V)

    updates = {model.U: model.U - learning_rate * g_U, \
               model.V: model.V - learning_rate * g_V}

    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates, givens={
                                    sparseData: s_d[index * batch_size: (index + 1) * batch_size],
                                    rowInd: r_d[index * batch_size: (index + 1) * batch_size],
                                    colInd: c_d[index * batch_size: (index + 1) * batch_size] })

    for i in range(50):
    	for minibatch_index in xrange(n_train_batches):
            avg_cost = train_model(minibatch_index)
            print str(i) + ":"+str(minibatch_index) + ' error: %4.4f' % avg_cost
    """

if __name__ == '__main__':
	main()