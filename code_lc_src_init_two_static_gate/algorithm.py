# learning algorithms
import numpy
import theano
import theano.tensor as T
from itertools import izip


def adadelta(parameters, gradients, rho=0.95, eps=1e-6):
    # create variables to store intermediate updates
    gradients_sq = [theano.shared(numpy.zeros(p.get_value().shape,
    							  dtype=theano.config.floatX))
    			    for p in parameters]
    deltas_sq = [theano.shared(numpy.zeros(p.get_value().shape,
    						   dtype=theano.config.floatX))
    		    for p in parameters]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [rho*g_sq + (1-rho)*(g**2)
    				   for g_sq,g in izip(gradients_sq, gradients)]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [(T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad
    		 for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients)]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas)]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
    #parameters_updates = [(p,T.clip(p - d, -15,15)) for p,d in izip(parameters,deltas)]
    parameters_updates = [(p,p-d) for p,d in izip(parameters,deltas)]

    return gradient_sq_updates + deltas_sq_updates + parameters_updates


def grad_clip(grads, clip_c):
    # apply gradient clipping
    if clip_c > 0:
        g2 = 0.

        for g in grads:
            g2 += (g**2).sum()

        new_grads = []
        for g in grads:
            new_grads.append(T.switch(g2 > (clip_c ** 2),
                             g / T.sqrt(g2) * clip_c,
                             g))
        grads = new_grads

    return grads

