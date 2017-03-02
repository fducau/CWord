from __future__ import print_function
from __future__ import absolute_import



import dateutil
import dateutil.tz
import datetime
import argparse
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from nmt import train


parser = argparse.ArgumentParser()
parser.add_argument('--dim_word', default=620,
                    help='Word dimension',
                    type=int)
parser.add_argument('--dim', default=1000,
                    help='Number of LSTM units',
                    type=int)
parser.add_argument('--encoder', default='gru',
                    help='Encoder type: supported gru')
parser.add_argument('--decoder', default='gru_cond',
                    help='Decoder type: supported gru_cond')
parser.add_argument('--patience', default=5,
                    help='Early stopping patience (in number of batches)',
                    type=int)
parser.add_argument('--maxEpochs', default=100,
                    help='Number of epochs.',
                    type=int)
parser.add_argument('--dispFreq', default=250,
                    help='Training display frequency',
                    type=int)
parser.add_argument('--nWords', default=20000,
                    help='Length of source vocabulary',
                    type=int)
parser.add_argument('--maxLen', default=30,
                    help='Maximum length for a sentence/utterance',
                    type=int)
parser.add_argument('--optimizer', default='adadelta',
                    help='Length of source vocabulary')
parser.add_argument('--batchSize', default=128,
                    help='Batch size',
                    type=int)
parser.add_argument('--validBatchSize', default=128,
                    help='Generate some samples in between sampleFreqs updates.',
                    type=int)
parser.add_argument('--saveModelTo', default='./ckt/',
                    help='Folder to save the model to.')
parser.add_argument('--validFreq', default=15000,
                    help='Number of batches in between validation steps',
                    type=int)
parser.add_argument('--saveFreq', default=5000,
                    help='Number of batches in between model savings.',
                    type=int)
parser.add_argument('--sampleFreq', default=1500,
                    help='Generate some samples in between sampleFreqs updates.',
                    type=int)
parser.add_argument('--dataset', default='data_iterator',
                    help='Type of data iterator and preprocessing to use')
parser.add_argument('--dictionary', default='',
                    help='Vocabulary dictionary to use')
parser.add_argument('--reload_', default='False',
                    help='Reload previous saved model?')
parser.add_argument('--dataset_', default='opensubs',
                    help='Dataset to be used')
parser.add_argument('--useContext', default='False',
                    help='Use context for word embededings?')
parser.add_argument('--dimContext', default=0,
                    help='Dimension of the context vector',
                    type=int)
parser.add_argument('--datasetSize', default=-1, type=int)



args = parser.parse_args()

dim_word = args.dim_word
dim = args.dim
encoder = args.encoder
decoder = args.decoder
patience = args.patience
max_epochs = args.maxEpochs
dispFreq = args.dispFreq
n_words_src = args.nWords
n_words = args.nWords
maxlen = args.maxLen
optimizer = args.optimizer
batch_size = args.batchSize
valid_batch_size = args.validBatchSize
saveto = args.saveModelTo
validFreq = args.validFreq
saveFreq = args.saveFreq
sampleFreq = args.sampleFreq
dataset = args.dataset
dictionary = args.dictionary
if args.reload_ == 'False':
    reload_ = False
else:
    reload_ = args.reload_

dataset_ = args.dataset_

if args.useContext == 'True':
    use_context = True
else:
    use_context = False

dim_context = args.dimContext
dataset_size = args.datasetSize
if dataset_ == 'opensubs':
    maxlen = 18


def main():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "./logs/"
    exp_name = "encDecAtt_%s" % timestamp

    train_err, valid_err, test_err = train(dim_word=dim_word,
                                           dim=dim,
                                           encoder=encoder,
                                           decoder=decoder,
                                           hiero=None,  # 'gru_hiero', # or None
                                           patience=patience,
                                           max_epochs=max_epochs,
                                           dispFreq=dispFreq,
                                           decay_c=0.,
                                           alpha_c=0.,
                                           diag_c=0.,
                                           lrate=0.01,
                                           n_words_src=n_words_src,
                                           n_words=n_words,
                                           maxlen=maxlen,
                                           optimizer=optimizer,
                                           batch_size=batch_size,
                                           valid_batch_size=valid_batch_size,
                                           saveto=saveto,
                                           validFreq=validFreq,
                                           saveFreq=saveFreq,
                                           sampleFreq=sampleFreq,
                                           dataset=dataset,
                                           dictionary=dictionary,
                                           dictionary_src=dictionary,
                                           use_dropout=False,
                                           reload_=reload_,
                                           correlation_coeff=0.1,
                                           clip_c=1., 
                                           dataset_=dataset_,
                                           use_context=use_context, 
                                           dim_context=dim_context, 
                                           dataset_size=dataset_size)


if __name__ == '__main__':
    main()
