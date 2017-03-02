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

raw_file_src = "raw.src"
raw_file_tgt = 'raw.tgt'
inc = 0


def main():
    parser = argparse.ArgumentParser(description='Set parameters for xml parser.')
    parser.add_argument('--rootXmlDir',
                        default="OpenSubtitles/en/",
                        help='Path to root directory of xml files')
    parser.add_argument('--dataDir',
                        default="data/",
                        help='Path to directory process data will be saved.')
    args = parser.parse_args()

    processed_data_dir = args.dataDir
    raw_data_dir = args.rootXmlDir

    files = findXmlFiles(raw_data_dir)
    print "Have " + str(len(files)) + " to parse!"
    # Setup folder structure and data file
    mkdir_p(processed_data_dir)
    for f in files:
        try:
            extractTokenizedPhrases(f, processed_data_dir)
        except KeyboardInterrupt:
            print "Process stopped by user..."
            return 0
        except Exception as e:
            print e
            print "Error in " + f
            pass


'''
Loops through folders recursively to find all xml files
'''


def findXmlFiles(directory):
    # Only gets one xml file per directory.
    # Used because of multiple subtitles of the same movie per folder.
    xmlFiles = []
    for f in os.listdir(directory):
        if os.path.isdir(directory + f):
            xmlFiles = xmlFiles + findXmlFiles(directory + f + "/")
        else:
            if not xmlFiles:
                xmlFiles.append(directory + f)

    return xmlFiles


'''
The assumption is made (for now) that each <s> node in the xml docs represents
a token, meaning everything has already been tokenized. At first observation
this appears to be an ok assumption.

This function has been modified to print to a single file for each movie
This is for memory consideration when processing later down the pipeline
'''


def extractTokenizedPhrases(xmlFilePath, dataDirFilePath):
    global inc
    inc += 1

    mkfile(dataDirFilePath + str(inc) + raw_file_src)

    tree = ET.parse(xmlFilePath)
    root = tree.getroot()

    print "Processing " + xmlFilePath + "..."
    prev_time = datetime.strptime("00:00:00,000", '%H:%M:%S,%f')
    A = []

    for child in root.findall('s'):
        for node in child.getiterator():

            if node.tag == 'time':
                items = node.items()
                id_item = items[0][1]
                time_item = items[1][1]
                if id_item[-1] == 'S':
                    ctime = datetime.strptime(time_item.encode('ascii', 'ignore'),
                                              '%H:%M:%S,%f')

                    if (ctime - prev_time) > timedelta(0, 60):
                        # Conversation ends __eod__= end of dialog
                        A.append("__eod__" + "\n")

                    prev_time = ctime

                elif id_item[-1] == 'E':
                    A.append('__eou__')
                    A.append('__eot__')

            if node.tag == 'w':
                    A.append(node.text.encode('ascii', 'ignore').replace('-', ''))

        if A[-1] != '__eot__':
            A.append('__eou__')

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
