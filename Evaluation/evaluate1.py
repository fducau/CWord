# -*- coding: utf-8 -*-
# @Author: fducau
# @Date:   2016-11-23 13:50:50
# @Last Modified by:   fducau
# @Last Modified time: 2016-12-13 00:08:38

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

from scipy import optimize, stats
from collections import OrderedDict

import wmt14enfr
import iwslt14zhen
import openmt15zhen
import trans_enhi
import stan
import data_iterator

import dateutil
import dateutil.tz
import datetime

from nmt import *
from nltk.translate.bleu_score import corpus_bleu

profile = False

def load_lines(input_file):
    x = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            line_w_eos = line + ' 0'
            source = numpy.array(map(int, line_w_eos.split()), dtype=numpy.int64)

            x.append(source)
    return x


def evaluate(dim_word=620,  # word vector dimensionality
             dim=1000,  # the number of LSTM units
             encoder='gru',
             decoder='gru_cond',
             hiero=None,  # 'gru_hiero', # or None
             patience=10,
             max_epochs=100,
             dispFreq=500,
             decay_c=0.,
             alpha_c=0.,
             diag_c=0.,
             lrate=0.01,
             n_words_src=20000,
             n_words=20000,
             maxlen=30,  # maximum length of the description
             optimizer='adadelta',
             batch_size=128,
             valid_batch_size=128,  # Validation and test batch size
             saveto='./ckt/',
             validFreq=6600,
             saveFreq=6600,  # save the parameters after every saveFreq updates
             sampleFreq=1500,  # generate some samples after every sampleFreq updates
             dataset='data_iterator',
             dictionary='',  # word dictionary
             dictionary_src='',  # word dictionary
             use_dropout=False,
             reload_=True,
             correlation_coeff=0.1,
             clip_c=1.,
             dataset_='opensubs', 
             use_context=False, 
             dim_context=0, 
             dataset_size=-1,
             perplexity=False,
             BLEU=True):

    # Model options
    model_options = locals().copy()
    # Reload previous saved options
    if reload_:
        with open('{}.npz.pkl'.format(reload_), 'rb') as f:
            model_options = pkl.load(f)

    if model_options['dictionary']:
        with open(model_options['dictionary'], 'rb') as f:
            word_dict = pkl.load(f)
    else:
        # Assume dictionary is in the same folder as data
        if model_options['dataset_'] == 'opensubs':
            dictionary = './data/OpenSubsDS/source_train_dict.pkl'
        elif model_options['dataset_'] == 'ubuntu':
            dictionary = './data/UbuntuDS/source_train_dict.pkl'
        else:
            raise ValueError('No dictionary specified.')

        with open(dictionary, 'rb') as f:
            word_dict = pkl.load(f)

    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk


    if model_options['dictionary_src']:
        with open(model_options['dictionary_src'], 'rb') as f:
            word_dict_src = pkl.load(f)
    else:
        # Assume dictionary is in the same folder as data
        if model_options['dataset_'] == 'opensubs':
            dictionary_src = './data/OpenSubsDS/source_train_dict.pkl'
        elif model_options['dataset_'] == 'ubuntu':
            dictionary_src = './data/UbuntuDS/source_train_dict.pkl'
        else:
            raise ValueError('No dictionary specified.')

        with open(dictionary_src, 'rb') as f:
            word_dict_src = pkl.load(f)

    word_idict_src = dict()
    for kk, vv in word_dict_src.iteritems():
        word_idict_src[vv] = kk


    print 'Loading data...'
    load_data, prepare_data = get_dataset(model_options['dataset'])

    if model_options['dataset_'] == 'opensubs':
        train, valid, test = load_data(train_batch_size=model_options['batch_size'],
                                       val_batch_size=model_options['valid_batch_size'],
                                       test_batch_size=model_options['valid_batch_size'],
                                       use_context=model_options['use_context'],
                                       dataset_size=model_options['dataset_size'])
    elif model_options['dataset_'] == 'ubuntu':
        train, valid, test = load_data(train_source_path='./data/UbuntuDS/source_train_idx',
                                       train_target_path='./data/UbuntuDS/target_train_idx',
                                       validation_source_path='./data/UbuntuDS/source_val_idx',
                                       validation_target_path='./data/UbuntuDS/target_val_idx',
                                       test_source_path='./data/UbuntuDS/source_test_idx',
                                       test_target_path='./data/UbuntuDS/target_test_idx',
                                       train_batch_size=model_options['batch_size'],
                                       val_batch_size=model_options['valid_batch_size'],
                                       test_batch_size=model_options['valid_batch_size'],
                                       use_context=model_options['use_context'],
                                       context_path={'train': './data/UbuntuDS/context_train_idx',
                                                     'validation': './data/UbuntuDS/context_val_idx',
                                                     'test': './data/UbuntuDS/context_test_idx'},
                                       dataset_size=model_options['dataset_size'])

    print 'Building model...'
    params = init_params(model_options)
    # reload parameters
    if reload_:
        params = load_params(reload_, params)

    tparams = init_tparams(params)

    trng, use_noise, x, x_mask, y, y_mask, conv_context, conv_context_mask, opt_ret, cost = build_model(tparams, model_options)
    print 'Use context: {}'.format(use_context)
    if model_options['use_context']:
        inps = [x, x_mask, y, y_mask, conv_context, conv_context_mask]
    else:
        inps = [x, x_mask, y, y_mask]

    print 'Buliding sampler...'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # Before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    if model_options['decay_c'] > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    if model_options['alpha_c'] > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(model_options['alpha_c']), name='alpha_c')
        alpha_reg = alpha_c * ((tensor.cast(y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
                                opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    print 'Building f_grad...',
    f_grad = theano.function(inps, grads, profile=profile)
    print 'Done'

    # Cliping gradients
    if model_options['clip_c'] > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    f_update = eval(model_options['optimizer'])(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_:
        history_errs = list(numpy.load('{}.npz'.format(reload_))['history_errs'])
    best_p = None
    bad_count = 0

    uidx = 0
    estop = False
    save_turn = 0
    ########################
    # Main evaluation loop
    ########################
    if perplexity:
        #print('Evaluating on train')
        #train_err, train_perplexity = prediction_scores(f_log_probs,
        #                                                prepare_data,
        #                                                model_options,
        #                                                train)
        #print('Train Cost: {} Train Perplexity: {}'.format(train_err, train_perplexity))

        print('Evaluating on validation')
        valid_err, valid_perplexity = prediction_scores(f_log_probs,
                                                        prepare_data,
                                                        model_options,
                                                        valid)
        print('Valid Cost: {} Valid Perplexity: {}'.format(valid_err, valid_perplexity))

        print('Evaluating on test')
        test_err, test_perplexity = prediction_scores(f_log_probs,
                                                      prepare_data,
                                                      model_options,
                                                      test)
        print('Test Cost: {} Test Perplexity: {}'.format(test_err, test_perplexity))

    stochastic = False
    if BLEU:
        references = []
        hypotheses = []
        print('Computing BLEU in validation set...')
        for x, y, conv_context in valid:
            x, x_mask, y, y_mask, conv_context, conv_context_mask = prepare_data(x, y, conv_context,
                                                                                 maxlen=maxlen,
                                                                                 n_words_src=n_words_src,
                                                                                 n_words=n_words)
            for utterance_idx in xrange(y.shape[1]):
                utterance = y[:, utterance_idx]
                m = y_mask[:, utterance_idx]

                utterance = utterance[m.nonzero()]
                references.append([[str(i) for i in utterance]])

            for utterance_idx in xrange(x.shape[1]):
                sample, score = gen_sample(tparams,
                                           f_init,
                                           f_next,
                                           x[:, utterance_idx][:, None],
                                           model_options,
                                           trng=trng, k=1,
                                           maxlen=30,
                                           stochastic=stochastic,
                                           argmax=True)

                hypotheses.append([str(i) for i in sample[0]])


        valid_BLEU = corpus_bleu(references, hypotheses)
        print('Validation BLEU: {}'.format(valid_BLEU))

        references = []
        hypotheses = []
        print('Computing BLEU in test set...')
        for x, y, conv_context in test:
            x, x_mask, y, y_mask, conv_context, conv_context_mask = prepare_data(x, y, conv_context,
                                                                                 maxlen=maxlen,
                                                                                 n_words_src=n_words_src,
                                                                                 n_words=n_words)
            for utterance_idx in xrange(y.shape[1]):
                utterance = y[:, utterance_idx]
                m = y_mask[:, utterance_idx]

                utterance = utterance[m.nonzero()]
                references.append([[str(i) for i in utterance]])

            for utterance_idx in xrange(x.shape[1]):
                sample, score = gen_sample(tparams,
                                           f_init,
                                           f_next,
                                           x[:, utterance_idx][:, None],
                                           model_options,
                                           trng=trng, k=1,
                                           maxlen=30,
                                           stochastic=stochastic,
                                           argmax=True)

                hypotheses.append([str(i) for i in sample[0]])

        test_BLEU = corpus_bleu(references, hypotheses)
        print('Test BLEU: {}'.format(test_BLEU))


if __name__ == '__main__':
    # evaluate(reload_='./evaluating/ctx001_2M_turn_1')
    print 'Loading OpenSubs CTX 2M'
    evaluate(reload_='./ckt_opensubs/ctx001_2M_turn_1')
    print 'Loading Ubuntu ATT'
    evaluate(reload_='./ckt_ubuntu/att001_turn_0')
    print 'Loading Ubuntu ctx'
    evaluate(reload_='./ckt_ubuntu/ctx001_2M_turn_1')
    print 'Loading OpenSubs ATT 2M'
    evaluate(reload_='./ckt_opensubs/att_006_2M_turn_0')





# Sample
# [[2324, 1, 3734, 1, 1, 3, 0]]
# X
# [[  377   146   207   207   207   207  5210]
#  [   25     6   226     7     6  1664   120]
#  [    5     4   162    59    39   116   614]
#  [  359     3     7    40  9634    32  4041]
#  [    4     0    84     8    10  9634  2182]
#  [    3     0     6     1    40   421 12080]
#  [   43     0   260   148   685     6     3]
#  [    7     0    22    10    47 17613     0]
#  [  153     0     3   607     6    10     0]
#  [    9     0     0    32    39     8     0]
#  [    3     0     0  1347   357  2118     0]
#  [   37     0     0     3    21   207     0]
#  [    4     0     0     0   207     3     0]
#  [    3     0     0     0     3     0     0]
#  [    0     0     0     0     0     0     0]]
# Y
# [[  146   207   207   207   207  5210  2324]
#  [    6   226     7     6  1664   120     1]
#  [    4   162    59    39   116   614  3734]
#  [    3     7    40  9634    32  4041     1]
#  [    0    84     8    10  9634  2182     3]
#  [    0     6     1    40   421 12080     0]
#  [    0   260   148   685     6     3     0]
#  [    0    22    10    47 17613     0     0]
#  [    0     3   607     6    10     0     0]
#  [    0     0    32    39     8     0     0]
#  [    0     0  1347   357  2118     0     0]
#  [    0     0     3    21   207     0     0]
#  [    0     0     0   207     3     0     0]
#  [    0     0     0     3     0     0     0]
#  [    0     0     0     0     0     0     0]]
# 