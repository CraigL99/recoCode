""" Super-simple Matrix factorization (of a random matrix) by Gradient
descent."""

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from time import sleep
import scipy.sparse as sp
from theano import sparse

plt.ion()  # interactive plotting

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

    shared_sparseData = theano.shared( np.asarray(sparseData, dtype=theano.config.floatX) )
    shared_rowInd = theano.shared( np.asarray(rowInd, dtype=theano.config.floatX) )
    shared_colInd = theano.shared( np.asarray(colInd, dtype=theano.config.floatX) )

    return shared_sparseData, T.cast(shared_rowInd, 'int32'), T.cast(shared_colInd, 'int32'), actualMatrix

def main():
    sparseData = T.vector('sparseData', dtype=theano.config.floatX)
    rowInd = T.vector('rowInd', dtype='int32')  # symbolic var for row indices
    colInd = T.vector('colInd', dtype='int32')  # symbolic var for col indices

    index = T.lscalar()  # index to a [mini]batch
    learning_rate = 0.0001
    n_users = 943
    n_items = 1682
    n_fac = 30
    batch_size = 10000 #must be a multiple of the total number of ratings

    model = MatrixFactorization(n_users=n_users, n_items=n_items, n_fac=n_fac)

    #evaluate_model = theano.function(inputs=[sparseData, rowInd,colInd ], outputs=model.squared_error(sparseData, rowInd, colInd))

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

    n_train_batches = s_d.get_value().shape[0] / batch_size
    n_test_batches = s_d_test.get_value().shape[0] / batch_size
    n_valid_batches = s_d_validate.get_value().shape[0] / batch_size 

    print "Number of ratings: " + str(s_d.get_value().shape[0])
    print "Batch size: " + str(batch_size)
    print "n_train_batchs: " + str(n_train_batches)
    print "n_test_batchs: " + str(n_test_batches)
    print "n_valid_batchs: " + str(n_valid_batches)


    # create a matrix for use in the visual representation
    #sparse_coo = sp.coo_matrix((np.cast[np.int](s_d.get_value()), (np.cast[np.int](r_d.get_value()), np.cast[np.int](c_d.get_value()))), shape=(n_users, n_items))
    #actualMatrix = sparse_coo.todense()
    #actualMatrix = ([1,2], [3,4]) #//TODO change back to actual matrix

    model.U.set_value(0.01 * np.random.randn(n_users, n_fac))
    model.V.set_value(0.01 * np.random.randn(n_fac, n_items))

    #Regularization 
    lambda_L2 = 0.0001
    lambda_L1 = 0.0000
    L2 = T.sum(s_d ** 2)
    L1 = T.sum(abs(s_d))

    cost = model.squared_error(sparseData, rowInd, colInd) + L2 * lambda_L2 + L1 * lambda_L1

    g_U = T.grad(cost=cost, wrt=model.U)
    g_V = T.grad(cost=cost, wrt=model.V)

    #updates is a dictionary, U and V are keys to the values...
    updates = {model.U: model.U - learning_rate * g_U, \
               model.V: model.V - learning_rate * g_V}

    #have to replace inputs as an index representing the minibatch
    # add givens parameter, to replace references to sparseData, rowInd, colInd with data slices from
    #   shared variables, see example online
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates, givens={
                                    sparseData: s_d[index * batch_size: (index + 1) * batch_size],
                                    rowInd: r_d[index * batch_size: (index + 1) * batch_size],
                                    colInd: c_d[index * batch_size: (index + 1) * batch_size] })

    # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
    test_model = theano.function(inputs=[index],
                                  outputs=model.errors(sparseData, rowInd, colInd),
                                  givens={
                                    sparseData: s_d_test[index * batch_size: (index + 1) * batch_size],
                                    rowInd: r_d_test[index * batch_size: (index + 1) * batch_size],
                                    colInd: c_d_test[index * batch_size: (index + 1) * batch_size] })

    validate_model = theano.function(inputs=[index],
                                  outputs=model.errors(sparseData, rowInd, colInd),
                                  givens={
                                    sparseData: s_d_validate[index * batch_size: (index + 1) * batch_size],
                                    rowInd: r_d_validate[index * batch_size: (index + 1) * batch_size],
                                    colInd: c_d_validate[index * batch_size: (index + 1) * batch_size] })


    fig, subs = plt.subplots(nrows=1, ncols=2, num=1)  # num is figure number
    canvas = fig.canvas
    plt.show()

    # Note: subs shape is (n_rows, n_cols)
    # But it is 1d is n_rows or n_cols = 1

    imargs = {'cmap': 'gray', 'interpolation': 'nearest'}

    subs[0].set_title('Actual Matrix')
    im0 = subs[0].imshow(actualMatrix, animated=True, **imargs)

    subs[1].set_title('Approximation by factoring')
    im1 = subs[1].imshow(np.dot(model.U.get_value(), model.V.get_value()),
                   animated=True, **imargs)

    for s in subs:
        s.set_axis_off()

    # note that all of this with the canvas, bounding boxes, blitting, etc
    # is to do a more efficient animation
    # repeated calls to imshow is very inefficient;
    # it's better to simply update the underlying array and
    # blit only the part of the canvas that has changed
    plt.tight_layout(pad=0.1)
    canvas.draw()
    bg = (canvas.copy_from_bbox(subs[0].bbox),
          canvas.copy_from_bbox(subs[1].bbox))
    sleep(1)

    #early stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                      # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
    best_validation_loss = np.inf
    validation_frequency = n_train_batches #Go through this many minibatches before checking on validation set
    best_params = None
    test_score = 0.
    done_looping = False
    for i in range(50):
        if (done_looping):
            break
        for minibatch_index in xrange(n_train_batches):
            avg_cost = train_model(minibatch_index)
            print str(i) + ":"+str(minibatch_index) + ' error: %4.4f' % avg_cost

            iter = (i * n_train_batches) + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_error = [ validate_model(g)
                                    for g in xrange(n_valid_batches) ]
                this_validation_error = np.mean(validation_error)
                print "Validation_error: " + str(this_validation_error)
                
                #if we got the best validation score until now
                if this_validation_error < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_error < best_validation_loss * \
                        improvement_threshold:
                        print '*****significant improvement****'
                        patience = max(patience, iter*patience_increase)
                    
                    best_validation_loss = this_validation_error
                    #test it on the test set
                    test_error = [ test_model(g)
                                    for g in xrange(n_test_batches) ]
                    test_score = np.mean(validation_error)
                    print ('test error of best model: %lf' % (test_score))
            if patience <= iter:
                done_looping = True
                break

        approx = np.dot(model.U.get_value(), model.V.get_value())

        fig.canvas.restore_region(bg[1])
        im1.set_array(approx)

        # scale is changing in approximation
        # so must renormalize
        # otherwise image will saturate
        im1.set_norm(mc.Normalize())
        subs[1].draw_artist(im1)

        fig.canvas.blit(im1.axes.bbox)

        # sleep(1.0/15)  # slow down the plotting

if __name__ == '__main__':
    main()
    
