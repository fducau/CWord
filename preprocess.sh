#!/bin/bash

echo 'Processing datasets...'


# Process train dataset and create vocabulary dict
echo 'Processing train...'
python preprocess.py --createDict=False --inputFile=./data/OpenSubsDS/source_train
python preprocess.py --createDict=False --inputFile=./data/OpenSubsDS/target_train
python preprocess.py --createDict=False --inputFile=./data/OpenSubsDS/context_train

echo 'Processing val...'
python preprocess.py --createDict=False --inputFile=./data/OpenSubsDS/source_val
python preprocess.py --createDict=False --inputFile=./data/OpenSubsDS/target_val
python preprocess.py --createDict=False --inputFile=./data/OpenSubsDS/context_val

echo 'Processing test...'
python preprocess.py --createDict=False --inputFile=./data/OpenSubsDS/source_test
python preprocess.py --createDict=False --inputFile=./data/OpenSubsDS/target_test
python preprocess.py --createDict=False --inputFile=./data/OpenSubsDS/context_test
