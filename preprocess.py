import sys
import numpy
import argparse
import cPickle as pkl
from collections import OrderedDict
import os


parser = argparse.ArgumentParser()
parser.add_argument('--createDict', default='False', help='Create vocabulary from file')
parser.add_argument('--inputFile', help='Dataset to read')
args = parser.parse_args()

def main():

    if args.createDict == 'True':
        create_dict = True
    else:
        create_dict = False

    input_file = args.inputFile
    if create_dict:
        print('Creating word dictionary...')

        word_freqs = OrderedDict()

        # Calculate word frequencies
        with open(input_file, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

        words = word_freqs.keys()
        freqs = word_freqs.values()

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['eos'] = 0
        worddict['UNK'] = 1
        worddict['__nc__'] = 2 # Used for empty lines for context. Means no context

        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii + 3 # Start counting from the first non reserved word

        with open('{}_dict.pkl'.format(input_file), 'wb') as f:
            pkl.dump(worddict, f)

    # Get indexed version of dataset
    print('Processing dataset...')

    dict_folder = '/'.join(input_file.split('/')[:-1]) + '/'

    files = os.listdir(dict_folder)
    for f in files:
        if 'dict.pkl' in f:
            dict_file = f
            break
    else:
        raise ValueError('Dictionary not found.')

    worddict = pkl.load(open(dict_folder + dict_file))

    indexed_dataset = []
    with open(input_file, 'r') as f:
        for line in f:
            sentence = []
            words_in = line.strip().split(' ')
            if len(words_in) == 1 and words_in[0] == '':
                words_in[0] = '__nc__'
            for w in words_in:
                sentence.append(str(worddict.get(w, '1')))

            indexed_dataset.append(sentence)

    with open('{}_idx'.format(input_file), 'wb') as f:
        for line in indexed_dataset:
            line_str = ' '.join(line)
            f.write(line_str + '\n')

    print 'Done'


if __name__ == '__main__':
    main()
