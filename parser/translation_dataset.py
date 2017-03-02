import os
import argparse
import numpy as np


def get_files(directory):
    files = []
    for f in os.listdir(directory):
        files.append(f)

    return files


def process_file(filename):
    source = []
    target = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            turns = line.strip()
            turns = turns.split('__eot__')

            for i in range(len(turns)):
                turns[i] = turns[i].strip()

            # last line is empty, ignoring it
            source = source + turns[:-2]
            target = target + turns[1:-1]

    return source, target

def process_context(filename):
    context = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            turns = line.strip()
            turns = turns.split('__eot__')

            for i in range(len(turns)):
                turns[i] = turns[i].strip()

            # last line is empty, ignoring it
            current_context = [" ".join(turns[max(0, i-3): i]) for i in range(0, len(turns)-2)]
            context = context + current_context
    return context

def randomize_file(path, permutation):
    orig = open(path, 'r')
    randomized = open(path + '_R', 'w')

    for line in orig.readlines():
        randomized.write(lines[permutation])


def randomize_data(output_dir, gen_ctx):
    source_file_train = open(output_dir + 'source_train', 'r')
    length = len(source_file_train.readlines())
    source_file_train.close()

    np.random.seed(0)
    permutation = np.random.permutation(np.arange(length))
    randomize_file(output_dir + 'source_train', permutation)
    randomize_file(output_dir + 'target_train', permutation)
    if gen_ctx:
        randomize_file(output_dir + 'context_train', permutation)





def main():
    parser = argparse.ArgumentParser(description='Set parameters for xml parser.')
    parser.add_argument('--destDir', default="./translation_dataset/", help='Path to output directory')
    parser.add_argument('--dataDir', default='./Processed/', help='Path to directory process data is stored.')
    parser.add_argument('--trainPct', default='0.8', help='Size of the train set')
    parser.add_argument('--valPct', default='0.1', help='Size of the validation set')
    parser.add_argument('--genCtx', default='True', help='Indicate if context file should be generated')
    parser.add_argument('--randomize', default='True', help='Indicate if files should be randomized')

    args = parser.parse_args()

    directory = args.dataDir
    files = get_files(directory=directory)

    output_dir = args.destDir
    randomize = args.randomize

    gen_ctx = args.genCtx
    if gen_ctx == 'True':
        gen_ctx = True
    else:
        gen_ctx = False

    

    source_file_train = open(output_dir + 'source_train', 'w')
    target_file_train = open(output_dir + 'target_train', 'w')

    if gen_ctx:
        context_file_train = open(output_dir + 'context_train', 'w')
    
    source_file_val = open(output_dir + 'source_val', 'w')
    target_file_val = open(output_dir + 'target_val', 'w')
    if gen_ctx:
        context_file_val = open(output_dir + 'context_val', 'w')
    
    source_file_test = open(output_dir + 'source_test', 'w')
    target_file_test = open(output_dir + 'target_test', 'w')
    if gen_ctx:
        context_file_test = open(output_dir + 'context_test', 'w')

    train_pct = np.double(args.trainPct)
    val_pct = np.double(args.valPct)

    if (val_pct + train_pct) > 1:
        raise ValueError('valPct + trainPct has to be lower than 1')

    n_files = len(files)

    train_size = int(train_pct * n_files)
    val_size = int(val_pct * n_files)
    test_size = len(files) - val_size - train_size

    np.random.seed(10)
    np.random.shuffle(files)

    print("Processing train files...")
    for f in files[:train_size]:
        source, target = process_file(directory + f)
        source_file_train.write('\n'.join(source))
        target_file_train.write('\n'.join(target))
        if gen_ctx:
            context = process_context(directory + f)
            context_file_train.write('\n'.join(context))

    print("Processing validation files...")
    for f in files[train_size:train_size + val_size]:
        source, target = process_file(directory + f)
        source_file_val.write('\n'.join(source))
        target_file_val.write('\n'.join(target))
        if gen_ctx:
            context = process_context(directory + f)
            context_file_val.write('\n'.join(context))

    print("Processing test files...")
    for f in files[train_size + val_size:]:
        source, target = process_file(directory + f)
        source_file_test.write('\n'.join(source))
        target_file_test.write('\n'.join(target))
        if gen_ctx:
            context = process_context(directory + f)
            context_file_test.write('\n'.join(context))

    source_file_train.close()
    target_file_train.close()
    source_file_val.close()
    target_file_val.close()
    source_file_test.close()
    target_file_test.close()
    if gen_ctx:
        context_file_train.close()
        context_file_val.close()
        context_file_test.close()



    if randomize == 'True':
        print('Randomizing files')
        randomize_data(output_dir, gen_ctx)



if __name__ == '__main__':
    main()



