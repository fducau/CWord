import numpy as np


class BatchIterator():
    """
    An iterator over a dataset file, which converts each
    line of the file into an example.

    The option ``'load_line'`` is a function which, given
    a string (a line in the file) outputs an example.
    """

    def __init__(self, source_path, target_path, context_path=None, batch_size=80, dataset_size=-1):
        self.source_path = source_path
        self.target_path = target_path
        self.context_path = context_path
        self.batch_size = batch_size

        self.source_f = open(self.source_path)
        self.target_f = open(self.target_path)

        self.counter = 0
        self.dataset_size = dataset_size
        if self.dataset_size <= 0:
            self.dataset_size = np.inf

        if self.context_path is not None:
            self.context_f = open(self.context_path)
        else:
            self.context_f = None

    def __iter__(self):
        return self

    def load_line(self, source_line, target_line, context_line=None):
        source = np.array(map(int, source_line.split()), dtype=np.int64)
        target = np.array(map(int, target_line.split()), dtype=np.int64)

        if context_line is not None:
            context = np.array(map(int, context_line.split()), dtype=np.int64) 
            return source, target, context

        return source, target

    def next(self):
        batch_x = []
        batch_y = []
        if self.context_f is not None:
            batch_c = []

        for i in range(self.batch_size):
            self.counter += 1
            source_line = self.source_f.readline()
            target_line = self.target_f.readline()

            context_line = None
            if self.context_f is not None:
                context_line = self.context_f.readline()

            if not (source_line or target_line or context_line) or self.counter > self.dataset_size:  # if line is empty
                if i == 0:
                    self.source_f.close()
                    self.target_f.close()

                    if self.context_f is not None:
                        self.context_f.close()
                        self.context_f = open(self.context_path)

                    self.source_f = open(self.source_path)
                    self.target_f = open(self.target_path)

                    self.counter = 0
                    raise StopIteration()
                else:
                    break
                    # If the file ended or reached the maximum dataset size.
                    # Return what it has till now

            lines = self.load_line(source_line, target_line, context_line)

            batch_x += [lines[0]]
            batch_y += [lines[1]]
            if context_line is not None:
                batch_c += [lines[2]]

        if context_line is not None:
            return (batch_x, batch_y, batch_c)
        return (batch_x, batch_y, None)


def prepare_data(seqs_x, seqs_y,
                 seqs_ctx,
                 maxlen=None,
                 n_words_src=30000,
                 n_words=30000):

    ctx_maxlen = 3 * maxlen
    if seqs_ctx is None:
        seqs_ctx = [np.array([])] * len(seqs_x)

    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    lengths_ctx = [len(s) for s in seqs_ctx]

    if maxlen:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        new_seqs_ctx = []
        new_lengths_ctx = []

        for l_x, s_x, l_y, s_y, l_c, s_c in zip(lengths_x, seqs_x, lengths_y, seqs_y, lengths_ctx, seqs_ctx):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_seqs_ctx.append(s_c[-ctx_maxlen:])
                new_lengths_ctx.append(min(l_c, ctx_maxlen))

        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        lengths_ctx = new_lengths_ctx
        seqs_ctx = new_seqs_ctx

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1
    maxlen_y = np.max(lengths_y) + 1
    maxlen_ctx = np.max(lengths_ctx)

    x = np.zeros((maxlen_x, n_samples)).astype('int64')
    y = np.zeros((maxlen_y, n_samples)).astype('int64')
    ctx = np.zeros((maxlen_ctx, n_samples)).astype('int64')

    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
    ctx_mask = np.zeros((maxlen_ctx, n_samples)).astype('float32')

    for idx, [s_x, s_y, s_c] in enumerate(zip(seqs_x, seqs_y, seqs_ctx)):
        s_x[np.where(s_x >= n_words_src - 1)] = 1
        s_y[np.where(s_y >= n_words - 1)] = 1
        s_c[np.where(s_c >= n_words_src - 1)] = 1

        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx] + 1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.

        ctx[:lengths_ctx[idx], idx] = s_c
        ctx_mask[:lengths_ctx[idx], idx] = 1.

    return x, x_mask, y, y_mask, ctx, ctx_mask


def load_data(train_source_path='./data/OpenSubsDS/source_train_idx',
              train_target_path='./data/OpenSubsDS/target_train_idx',
              validation_source_path='./data/OpenSubsDS/source_val_idx',
              validation_target_path='./data/OpenSubsDS/target_val_idx',
              test_source_path='./data/OpenSubsDS/source_test_idx',
              test_target_path='./data/OpenSubsDS/target_test_idx',
              train_batch_size=80,
              val_batch_size=80,
              test_batch_size=80, 
              use_context=False, 
              context_path={'train':'./data/OpenSubsDS/context_train_idx',
                            'validation':'./data/OpenSubsDS/context_val_idx',
                            'test':'./data/OpenSubsDS/context_test_idx'}, 
              dataset_size=-1):

    #############
    # LOAD DATA #
    #############

    print '... initializing data iterators'
    validation_size = int(dataset_size * 0.1)
    test_size = int(dataset_size * 0.1)

    if not use_context:
        train = BatchIterator(source_path=train_source_path,
                              target_path=train_target_path,
                              batch_size=train_batch_size,
                              dataset_size=dataset_size)
        valid = BatchIterator(source_path=validation_source_path,
                              target_path=validation_target_path,
                              batch_size=val_batch_size,
                              dataset_size=validation_size)
        test = BatchIterator(source_path=test_source_path,
                             target_path=test_target_path,
                             batch_size=test_batch_size,
                             dataset_size=test_size)
    if use_context:
        train = BatchIterator(source_path=train_source_path,
                              target_path=train_target_path,
                              context_path=context_path['train'],
                              batch_size=train_batch_size, 
                              dataset_size=dataset_size)

        valid = BatchIterator(source_path=validation_source_path,
                              target_path=validation_target_path,
                              context_path=context_path['validation'],
                              batch_size=val_batch_size, 
                              dataset_size=validation_size)

        test = BatchIterator(source_path=test_source_path,
                             target_path=test_target_path,
                             context_path=context_path['test'],
                             batch_size=test_batch_size, 
                             dataset_size=test_size)

    return train, valid, test


