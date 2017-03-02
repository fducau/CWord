'''
Metrics to evaluate the models
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

import os
import warnings
import sys
import time

import dateutil
import dateutil.tz
import datetime

# https://github.com/ddahlmeier/neural_lm/blob/master/lbl.py

def pred_perplexity(f_log_probs, prepare_data, options, iterator, verbose=False):
    log_probs = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=100, n_words_src=options['n_words_src'], n_words=options['n_words'])
        
        if x == None:
            continue

        batch_log_probs = f_log_probs(x,x_mask,y,y_mask)
        for p in batch_log_probs:
            log_p.append(p)

        if verbose:
            print >>sys.stderr, '%d samples computed'%(n_done)

    perplexity_exponent = - np.mean(log_probs)
    return np.power(2.0, perplexity_exponent)


def perplexity_from_logprobs(log_probs):
	return np.power(2.0, - np.mean(log_probs))


