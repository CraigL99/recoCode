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

    def squared_error(self, input):
        return T.sum(T.sqr(input - self.pred))

X = T.matrix('X')
learning_rate = 0.01
n_users = 943
n_items = 1682
n_fac = 100

model = MatrixFactorization(n_users=n_users, n_items=n_items, n_fac=n_fac)

evaluate_model = theano.function(inputs=[X, ], outputs=model.squared_error(X))


data = np.random.randn(n_users, n_items).astype(theano.config.floatX)

model.U.set_value(0.01 * np.random.randn(n_users, n_fac))
model.V.set_value(0.01 * np.random.randn(n_fac, n_items))

# B = model.U.get_value()
# C = model.V.get_value()
# print np.sum(np.square(data - (np.dot(BY,C))))
# print evaluate_model(data)

#cost now contains a function which calculates the squared error based off of X

# Where the magic happens, ask Graham to walk through it, I don't understand how 
    # taking the derivative of the sum of squared errors with respect to U/v could possibly work let alone factor the matrix...
cost = model.squared_error(X)

#take the derivative of cost with respect to U or V.
g_U = T.grad(cost=cost, wrt=model.U)
g_V = T.grad(cost=cost, wrt=model.V)

#updates is a dictionary, U and V are keys to the values...
updates = {model.U: model.U - learning_rate * g_U, \
           model.V: model.V - learning_rate * g_V}

train_model = theano.function(inputs=[X, ],
                              outputs=cost,
                              updates=updates)

#get_grads = theano.function(inputs=[X, ], outputs=[g_U, g_V])

fig, subs = plt.subplots(nrows=1, ncols=2, num=1)  # num is figure number
canvas = fig.canvas
plt.show()

# Note: subs shape is (n_rows, n_cols)
# But it is 1d is n_rows or n_cols = 1

imargs = {'cmap': 'gray', 'interpolation': 'nearest'}

subs[0].set_title('Matrix')
im0 = subs[0].imshow(data, animated=True, **imargs)

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
sleep(5)
for i in range(100):
    avg_cost = train_model(data)
    #print model.U.get_value()
    #print get_grads(data)
    print 'error: %4.4f' % avg_cost
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

    
