# -*- coding: utf-8 -*-
# @Author: fducau
# @Date:   2016-11-23 13:50:50
# @Last Modified by:   fducau
# @Last Modified time: 2016-12-07 18:24:58

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
             hiero=None,
             decay_c=0.,
             alpha_c=0.,
             diag_c=0.,
             lrate=0.01,
             n_words_src=20000,
             n_words=20000,
             maxlen=100,  # maximum length of the description
             optimizer='adadelta',
             batch_size=128,
             valid_batch_size=128,  # Validation and test batch size
             saveto='./ckt/',
             dataset='data_iterator',
             dictionary='',  # word dictionary
             dictionary_src='',  # word dictionary
             use_dropout=False,
             model=False,
             correlation_coeff=0.1,
             clip_c=1.,
             dataset_='opensubs',
             use_context=False,
             dataset_size=-1,
             perplexity=True,
             BLEU=True):

    model_options = locals().copy()
        # Reload previous saved options
    if model:
        with open('{}.npz.pkl'.format(model), 'rb') as f:
            model_options = pkl.load(f)
            for k, v in model_options.items():
                if (k == 'dim_word' or k=='dim' or k=='encoder' or k=='decoder' or
                k == 'n_words_src' or k=='n_words' or k=='optimizer' or k=='dataset' or
                k == 'dictionary' or k=='dictionary_src' or k=='dataset_' or
                k == 'use_context' or k=='dim_context' or k=='dataset_size'):
                    locals()[k] = v

                if k not in locals().keys():
                    locals()[k] = v
    else:
        raise ValueError('No model specified')


    # ===================
    # LOAD DICTIONARIES
    # ===================
    if dictionary:
        with open(dictionary, 'rb') as f:
            word_dict = pkl.load(f)
    else:
        # Assume dictionary is in the same folder as data
        if dataset_ == 'opensubs':
            dictionary = './data/OpenSubsDS/source_train_dict.pkl'
        elif dataset_ == 'ubuntu':
            dictionary = './data/UbuntuDS/source_train_dict.pkl'
        else:
            raise ValueError('No dictionary specified.')

        with open(dictionary, 'rb') as f:
            word_dict = pkl.load(f)

    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk

    if dictionary_src:
        with open(dictionary_src, 'rb') as f:
            word_dict_src = pkl.load(f)
    else:
        # Assume dictionary is in the same folder as data
        if dataset_ == 'opensubs':
            dictionary_src = './data/OpenSubsDS/source_train_dict.pkl'
        elif dataset_ == 'ubuntu':
            dictionary_src = './data/UbuntuDS/source_train_dict.pkl'
        else:
            raise ValueError('No dictionary specified.')

        with open(dictionary_src, 'rb') as f:
            word_dict_src = pkl.load(f)

    word_idict_src = dict()
    for kk, vv in word_dict_src.iteritems():
        word_idict_src[vv] = kk

    # =======================
    # LOAD MODEL PARAMETERS
    # =======================

    print 'Loading data...'
    load_data, prepare_data = get_dataset(dataset)

    if dataset_ == 'opensubs':
        train, valid, test = load_data(train_batch_size=batch_size,
                                       val_batch_size=valid_batch_size,
                                       test_batch_size=valid_batch_size,
                                       use_context=use_context,
                                       dataset_size=dataset_size)
    elif dataset_ == 'ubuntu':
        train, valid, test = load_data(train_source_path='./data/UbuntuDS/source_train_idx',
                                       train_target_path='./data/UbuntuDS/target_train_idx',
                                       validation_source_path='./data/UbuntuDS/source_val_idx',
                                       validation_target_path='./data/UbuntuDS/target_val_idx',
                                       test_source_path='./data/UbuntuDS/source_test_idx',
                                       test_target_path='./data/UbuntuDS/target_test_idx',
                                       train_batch_size=batch_size,
                                       val_batch_size=valid_batch_size,
                                       test_batch_size=valid_batch_size,
                                       use_context=use_context,
                                       context_path={'train': './data/UbuntuDS/context_train_idx',
                                                     'validation': './data/UbuntuDS/context_val_idx',
                                                     'test': './data/UbuntuDS/context_test_idx'},
                                       dataset_size=dataset_size)

    print 'Building model...'
    params = init_params(model_options)
    # reload parameters
    if model:
        params = load_params(model, params)
    else:
        raise ValueError('No model specified')

    tparams = init_tparams(params)

    trng, use_noise, x, x_mask, y, y_mask, conv_context, conv_context_mask, opt_ret, cost = build_model(tparams, model_options)

    if use_context:
        inps = [x, x_mask, y, y_mask, conv_context, conv_context_mask]
    else:
        inps = [x, x_mask, y, y_mask]

    # theano.printing.debugprint(cost.mean(), file=open('cost.txt', 'w'))

    print 'Buliding sampler...'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # Before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((tensor.cast(y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
                                opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    history_errs = []
    # reload history
    if model and os.path.exists(model):
        history_errs = list(numpy.load(model)['history_errs'])
    best_p = None
    bad_count = 0

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    uidx = 0
    estop = False
    save_turn = 0
    ########################
    # Main evaluation loop
    ########################
    if perplexity:
        print('Evaluating on train')
        # train_err, train_perplexity = prediction_scores(f_log_probs,
        #                                                 prepare_data,
        #                                                 model_options,
        #                                                 train)
        # print('Train Cost: {} Train Perplexity: {}'.format(train_err, train_perplexity))

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
        for x, y,  in valid:

            references.append([[str(i) for i in y]])

            sample, score = gen_sample(tparams, f_init, f_next, x[:, None],
                                       model_options, trng=trng, k=1, maxlen=30,
                                       stochastic=stochastic, argmax=True)

            hypotheses.append([str(i) for i in sample])

        valid_BLEU = corpus_bleu(references, hypotheses)
        print('Validation BLEU: '.format(valid_BLEU))

        references = []
        hypotheses = []

        for x, y, conv_context in test:

            references.append([[str(i) for i in y]])

            sample, score = gen_sample(tparams, f_init, f_next, x[:, None],
                                       model_options, trng=trng, k=1, maxlen=30,
                                       stochastic=stochastic, argmax=True)

            hypotheses.append([str(i) for i in sample])


        test_BLEU = corpus_bleu(references, hypotheses)
        print('Test BLEU: '.format(test_BLEU))

    for i, x in enumerate(source_utterances):
        stochastic = False
        sample, score = gen_sample(tparams, f_init, f_next, x[:, None],
                                   model_options, trng=trng, k=1, maxlen=30,
                                   stochastic=stochastic, argmax=True)

        print('Source {}: '.format(i) + print_utterance(x, word_idict))
        if stochastic:
            ss = sample
        else:
            score = score / numpy.array([len(s) for s in sample])
            ss = sample[score.argmin()]
        print('Sample {}:'.format(i) + print_utterance(ss, word_idict))


if __name__ == '__main__':
    evaluate(model='./ckt/ckt_opensubs/att006_2M/att_006_2M_turn_0')
