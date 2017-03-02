# -*- coding: utf-8 -*-
# @Author: fducau
# @Date:   2016-11-23 13:50:50
# @Last Modified by:   fducau
# @Last Modified time: 2016-12-19 04:14:49
import cPickle as pkl
import numpy as np
import argparse

from nmt import *

parser = argparse.ArgumentParser()

parser.add_argument('--model', default=None,
                    help='Path of model that will be loaded')
parser.add_argument('--test', default=None,
                    help='Load test conversation from the specified file, \
                    if no file is specifified activate interative mode')
parser.add_argument('--context', default=None,
                    help='Load test context from the specified file')
parser.add_argument('--output', default=None,
                    help='Output predictions in the specified file')
parser.add_argument('--dict', default=None,
                    help='Dictionary of words in the specified file')


args = parser.parse_args()
model_path = args.model
test_path = None
if not args.test == 'None':
    test_path = args.test

context_path = None
if not args.test == 'None':
    context_path = args.context

output_path = None
if not args.test == 'None':
    output_path = args.output

dict_path = None
if not args.dict == 'None':
    dict_path = args.dict


def load_lines(input_file):
    print 'input_file', input_file

    with open(input_file, 'r') as f:
        for line in f.readlines():
            line_w_eos = line + ' 0'
            source = np.array(map(int, line_w_eos.split()), dtype=np.int64)
            yield source

def replace_unkown(x, vocabulary_size):
    oov_mask = x >= vocabulary_size
    x[oov_mask] = 1
    return x


def generate_context_and_mask(conv_context, maxlen_context):
    len_mask_true = conv_context.shape[0]
    mask_true = np.ones(len_mask_true)
    len_mask_zeros = max(0, maxlen_context - len_mask_true)
    # print 'legth context_mask', len_mask_true + len_mask_zeros
    mask_zeros = np.zeros(len_mask_zeros)

    conv_context = np.concatenate((conv_context, mask_zeros)).astype(np.int64)
    conv_context_mask = np.concatenate((mask_true, mask_zeros)).astype(np.float32)
    return conv_context, conv_context_mask


def generate_target(input_file, model_options, tparams, word_idict, trng,
                    f_init, f_next, vocabulary_size,
                    output_file=None, k=5, stochastic=False):

    if output_file:
        f = open(output_file, 'w')

    for i, x in enumerate(load_lines(input_file)):
        # Replace unfrequent words with UNK
        x = replace_unkown(x, vocabulary_size)

        sample, score = gen_sample(tparams, f_init, f_next, x[:, None],
                                   model_options, trng=trng, k=k, maxlen=30,
                                   stochastic=stochastic, argmax=True)

        if stochastic:
            ss = sample
        else:
            score = score / np.array([len(s) for s in sample])
            ss = sample[score.argmin()]

        if output_file:
            f.write(print_utterance(ss, word_idict) + '\n')
        else:
            print('Source {}: '.format(i) + print_utterance(x, word_idict))
            print('Sample {}:'.format(i) + print_utterance(ss, word_idict))
        if i == 200000:
            break

    f.close()


def generate_target_context(input_file, context_file, model_options, tparams, word_idict, trng,
                            f_init, f_next, vocabulary_size,
                            maxlen_context, output_file=None, k=5, stochastic=False):

    if output_file:
        f = open(output_file, 'w')

    for i, (x, ctx) in enumerate(zip(load_lines(input_file), load_lines(context_file))):
        #print 'i', i, x, ctx
        x = replace_unkown(x, vocabulary_size)
        ctx = replace_unkown(ctx, vocabulary_size)
        conv_context = ctx[-maxlen_context:]
        conv_context, conv_context_mask = generate_context_and_mask(conv_context, maxlen_context)

        sample, score = gen_sample(tparams, f_init, f_next, x[:, None],
                                   model_options,
                                   conv_context[:, None],
                                   conv_context_mask[:, None, None],
                                   trng=trng, k=k, maxlen=30,
                                   stochastic=stochastic, argmax=True)

        if stochastic:
            ss = sample
        else:
            score = score / np.array([len(s) for s in sample])
            ss = sample[score.argmin()]

        if output_file:
            f.write(print_utterance(ss, word_idict) + '\n')
        else:
            print('Source {}: '.format(i) + print_utterance(x, word_idict))
            print('Sample {}:'.format(i) + print_utterance(ss, word_idict))
        if i % 100 == 0:
            print i
        if i == 200000:
            break

    f.close()


def chat(dim_word=620,  # word vector dimensionality
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
         dictionary='./data/OpenSubsDS/source_train_dict.pkl',  # word dictionary
         use_dropout=False,
         model=False,
         correlation_coeff=0.1,
         clip_c=1.,
         input_file='./chat_files/input_sentences.txt_idx',
         context_file='./chat_files/input_context.txt_idx',
         output_file='data/test_generation.txt'):

    # Model options
    model_options = locals().copy()

    if dictionary:
        with open(dictionary, 'rb') as f:
            word_dict = pkl.load(f)
        word_idict = dict()
        for kk, vv in word_dict.iteritems():
            word_idict[vv] = kk

    # Reload previous saved options
    if model:
        with open('{}.npz.pkl'.format(model), 'rb') as f:
            model_options = pkl.load(f)
    else:
        raise ValueError('No model specified')

    # Initializing variables from model_options
    vocabulary_size = model_options['n_words']
    use_context = model_options['use_context']
    maxlen_context = 3 * model_options['maxlen']
    context_window_size = 3
    current_context = [np.array([2, 0])]

    print 'Building model...'
    params = init_params(model_options)

    if model:
        params = load_params(model, params)

    tparams = init_tparams(params)

    print 'Model Options', model_options

    trng, use_noise, x, x_mask, y, y_mask, conv_context, conv_context_mask, opt_ret, cost = build_model(tparams, model_options)
    if use_context:
        inps = [x, x_mask, y, y_mask, conv_context, conv_context_mask]
    else:
        inps = [x, x_mask, y, y_mask]

    print 'Buliding sampler...'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    if input_file:  # Load test data from file and make predictions

        print 'use_context', use_context
        if use_context:
            generate_target_context(input_file, context_file, model_options, tparams, word_idict, trng,
                                    f_init, f_next, vocabulary_size,
                                    maxlen_context, output_file, k=5)
        else:
            generate_target(input_file, model_options, tparams, word_idict, trng,
                            f_init, f_next, vocabulary_size, output_file, k=5)
    else:  # Interactive mode
        while True:
            raw_utterance = raw_input("> ").lower()
            if raw_utterance == 'exit' or raw_utterance == 'quit':
                break
            splitted_utterance = raw_utterance.split()
            raw_utterance_ix = np.array([word_dict.get(word, 1) for word in splitted_utterance] + [0])
            oov_mask = raw_utterance_ix >= vocabulary_size
            raw_utterance_ix[oov_mask] = 1

            print raw_utterance_ix

            stochastic = False
            if use_context:
                if len(current_context) > context_window_size:
                    current_context = current_context[-context_window_size:]
                current_context.append(raw_utterance_ix)

                # Ignore last sentence and last word from sentence
                conv_context = [word for sentence in current_context[:-1] for word in sentence[:-1]]
                conv_context = np.array(conv_context)[-maxlen_context:]

                conv_context, conv_context_mask = generate_context_and_mask(conv_context, maxlen_context)

                print conv_context, conv_context_mask

                sample, score = gen_sample(tparams, f_init, f_next, raw_utterance_ix[:, None],
                                           model_options,
                                           conv_context[:, None],
                                           conv_context_mask[:, None, None],
                                           trng=trng, k=5, maxlen=30,
                                           stochastic=stochastic, argmax=True)

            else:
                sample, score = gen_sample(tparams, f_init, f_next, raw_utterance_ix[:, None],
                                           model_options, trng=trng, k=5, maxlen=30,
                                           stochastic=stochastic, argmax=True)

            if stochastic:
                ss = sample
            else:
                score = score / np.array([len(s) for s in sample])
                ss = sample[score.argmin()]

            for j, (samp, sc) in enumerate(zip(sample, score)):
                print j, sc, print_utterance(samp, word_idict)

            print 'ss', ss
            print('> ' + print_utterance(ss, word_idict))


if __name__ == '__main__':
    chat(model=model_path, input_file=test_path, 
        context_file=context_path, output_file=output_path,
        dictionary=dict_path)
