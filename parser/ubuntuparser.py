# @Author: fducau
# @Modified from: inikdom

'''
The point of this script is to parse all subtitle xml data for source target pairs
It will assume each line is the target of the previous line.
This will store the text data in a tokenized format, meant to be parsed by a deep learning
framework and put into a pre-processed data file.
'''

import xml.etree.ElementTree as ET
import argparse
import os
import re
import errno
from datetime import datetime
from datetime import timedelta
import pandas as pd

raw_file_src = "raw.src"
raw_file_tgt = 'raw.tgt'
inc = 0


def main():
    parser = argparse.ArgumentParser(description='Set parameters for xml parser.')
    parser.add_argument('--rootDir',
                        default="OpenSubtitles/en/",
                        help='Path to root directory of xml files')
    parser.add_argument('--dataDir',
                        default="data/",
                        help='Path to directory process data will be saved.')
    args = parser.parse_args()

    processed_data_dir = args.dataDir
    raw_data_dir = args.rootDir

    files = findTsvFiles(raw_data_dir)
    print "Have " + str(len(files)) + " to parse!"
    # Setup folder structure and data file
    mkdir_p(processed_data_dir)
    for f in files:
        try:
            extractTokenizedPhrases(f, processed_data_dir)
        except KeyboardInterrupt:
            print "Process stopped by user..."
            # return 0
        except Exception as e:
            print e
            print "Error in " + f
            pass


'''
Loops through folders recursively to find all xml files
'''


def findTsvFiles(directory):
    # Only gets one xml file per directory.
    # Used because of multiple subtitles of the same movie per folder.
    files = []
    for f in os.listdir(directory):
        if os.path.isdir(directory + f):
            files = files + findTsvFiles(directory + f + "/")
        else:
            #Load all the files in the folder
            files.append(directory + f)

    return files


'''
The assumption is made (for now) that each <s> node in the xml docs represents
a token, meaning everything has already been tokenized. At first observation
this appears to be an ok assumption.

This function has been modified to print to a single file for each movie
This is for memory consideration when processing later down the pipeline
'''


def extractTokenizedPhrases(tsvFilePath, dataDirFilePath):
    global inc
    inc += 1

    mkfile(dataDirFilePath + str(inc) + raw_file_src)

    ds = pd.read_csv(tsvFilePath, sep='\t', header=None)

    print "Processing " + tsvFilePath + "..."

    speaker_A = ds.ix[0][1]
    curr_speaker = speaker_A
    past_speaker = speaker_A

    A = []

    for i in ds.index:
        past_speaker = curr_speaker
        curr_speaker = ds.ix[i][1]

        if curr_speaker != past_speaker:
            A.append('__eot__')

        text = ds.ix[i][3]
        A.append(text.encode('ascii', 'ignore').replace('-', ''))

        A.append('__eou__')

    A.append('__eot__')
    A.append('__eod__')
    text = " ".join(A)
    text = cleanText(text)
    try:
        with open(dataDirFilePath + str(inc) + raw_file_src, 'a') as f:
            f.write(text + "\n")
    except IndexError:
        pass


'''
This function removes funky things in text
There is probably a much better way to do it, but unless the token list is
much bigger this shouldn't really matter how inefficient it is
'''

def cleanText(text):
    regex = re.compile('\(.+?\)')
    text = regex.sub('', text)
    regex = re.compile('\{.+?\}')
    text = regex.sub('', text)
    regex = re.compile('\[.+?\]')
    text = regex.sub('', text)

    text.replace("  ", " ")
    text = text.replace("~", "")
    text = text.strip(' ')
    text = text.lower()

    t = text.split('__eou__')
    t = [j for j in t if len(j) > 0]

    text_new = '__eou__'.join(t)

    t = text_new.split('\n')
    t1 = []
    for i in t:
        if i.strip() != '__eod__':
            t1.append(i)

    text_new = '\n'.join(t1)

    return text_new


'''
Taken from
http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
'''


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def mkfile(path):
    try:
        with open(path, 'w+'):
            return 1
    except IOError:
        print "Data file open, ensure it is closed, and re-run!"
        return 0


if __name__ == "__main__":
    main()
