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

    def squared_error(self, sparseData, IBM, UBM):
        UD = T.dot(UBM, self.U)
        VD = T.dot(IBM, (self.V).T)
        matrix = UD*VD
        pred = T.sum(matrix, axis=1)
        return T.sum(T.square(sparseData-pred))

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

#return a bitMatrix corresponding to the current batch indicated by index
def getUBM(index, batch_size, r_d, n_users):
    #create a matrix to hold user bit vectors, size 0..99 999 x 0..1681
    UBM = np.zeros((batch_size, n_users), dtype=theano.config.floatX)
    UBM[np.arange(batch_size), ((r_d.get_value()).astype(int)[index * batch_size: (index + 1) * batch_size])] = 1
    return UBM

def getIBM(index, batch_size, c_d, n_items):    
    #create a matrix to hold user bit vectors, size 0..99 999 x 0..942
    IBM = np.zeros((batch_size, n_items), dtype=theano.config.floatX)
    IBM[np.arange(batch_size), ((c_d.get_value()).astype(int)[index * batch_size: (index + 1) * batch_size])] = 1
    return IBM

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

    #compute number of baches
    n_train_batches = s_d.get_value().shape[0] / batch_size

    print "Number of ratings: " + str(s_d.get_value().shape[0])
    print "Batch size: " + str(batch_size)
    print "n_train_batchs: " + str(n_train_batches)

    model = MatrixFactorization(n_users=n_users, n_items=n_items, n_fac=n_fac)
    model.U.set_value(0.01 * np.random.randn(n_users, n_fac))
    model.V.set_value(0.01 * np.random.randn(n_fac, n_items))

    sparseData = T.vector('sparseData', dtype=theano.config.floatX)
    itemBitMatrix = T.matrix('itemBitMatrix', dtype=theano.config.floatX)
    userBitMatrix = T.matrix('itemBitMatrix', dtype=theano.config.floatX)

    cost = model.squared_error(sparseData, itemBitMatrix, userBitMatrix)

    g_U = T.grad(cost=cost, wrt=model.U)
    g_V = T.grad(cost=cost, wrt=model.V)

    updates = {model.U: model.U - learning_rate * g_U, \
               model.V: model.V - learning_rate * g_V}

    index = T.lscalar()
    train_model = theano.function(inputs=[sparseData, itemBitMatrix, userBitMatrix],
                                  outputs=cost,
                                  updates=updates)


    for i in xrange(50):
        for minibatch_index in xrange(n_train_batches):
            avg_cost = train_model(s_d.get_value()[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], getIBM(minibatch_index, batch_size, c_d, n_items), getUBM(minibatch_index, batch_size, r_d, n_users))
            print str(i) + ":"+str(minibatch_index) + ' error: %4.4f' % avg_cost

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