import numpy as np
import random
import linecache
import re
import csv
from collections import Counter
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
# from keras.preprocessing.sequence import pad_sequences


class Vocab:
    def __init__(self):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'

        self.word2id = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.id2word = {
            self.PAD: self.PAD_TOKEN,
            self.BOS: self.BOS_TOKEN,
            self.EOS: self.EOS_TOKEN,
            self.UNK: self.UNK_TOKEN,
        }

    def build_vocab(self, sentences, min_count=1, max_words=10000):
        tokenizer = Tokenizer(filters='', lower=True, char_level=False)
        tokenizer.fit_on_texts(sentences)
        for k, _ in Counter(tokenizer.word_counts).most_common(max_words):
            id = len(self.word2id)
            self.word2id[k] = id
            self.id2word[id] = k

        print("Total word2id: ", len(self.word2id))
        print("Total id2word: ", len(self.id2word))

        # self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter}

    def sentence_to_ids(self, sentence):
        return [self.word2id[word] if word in self.word2id else self.UNK for word in sentence]

def load_data(file_path, delim='\n'):
    '''
    Load entire text data from a file. We assume the file is a line-by-line text file
    # Arguments:
        file_path: str
    # Returns:
        data: list of sentences, data[i] means one line of text.
    '''
    data = []
    PUNCT = '!"#$%&()*+,-./:;<=>?\\[\\]\\\\^_`{|}~\\n\\t\'‘’“”’'
    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            # If training word level, must add spaces around each punctuation,
            # so that each punctuation can be treated as a word
            line = line.rstrip(delim)
            line = re.sub('(--|[{}])'.format(PUNCT), r' \1 ', line)
            line = re.sub(' {2,}', ' ', line)
            data.append(line)
    return data

def sentence_to_ids(vocab, sentence, UNK=3):
    '''
    # Arguments:
        vocab: SeqGAN.utils.Vocab
        sentence: list of str
    # Returns:
        ids: list of int
    '''
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    # ids += [EOS]
    return ids

def pad_seq(seq, max_length, PAD=0):
    """
    :param seq: list of int,
    :param max_length: int,
    :return seq: list of int,
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq

def print_ids(ids, vocab, verbose=True, exclude_mark=True, PAD=0, BOS=1, EOS=2):
    '''
    :param ids: list of int,
    :param vocab:
    :param verbose(optional): 
    :return sentence: list of str
    '''
    sentence = []
    for i, id in enumerate(ids):
        word = vocab.id2word[id]
        if exclude_mark and id == EOS:
            break
        if exclude_mark and id in (BOS, PAD):
            continue
        sentence.append(word)
    if verbose:
        print(sentence)
    return sentence


class GeneratorPretrainingFitGenerator(Sequence):
    '''
    Generate generator pretraining data.
    # Arguments
        path: str, path to data x
        B: int, batch size
        T (optional): int or None, default is None.
            if int, T is the max length of sequential data.
        min_count (optional): int, minimum of word frequency for building vocabrary
        shuffle (optional): bool

    # Params
        PAD, BOS, EOS, UNK: int, id
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN: str
        B, min_count: int
        vocab: Vocab
        word2id: Vocab.word2id
        id2word: Vocab.id2word
        raw_vocab: Vocab.raw_vocab
        V: the size of vocab
        n_data: the number of rows of data

    # Examples
        generator = VAESequenceGenerator('./data/train_x.txt', 32)
        x, y_true = generator.__getitem__(idx=11)
        print(x[0])
        >>> 8, 10, 6, 3, 2, 0, 0, ..., 0
        print(y_true[0][0])
        >>> 0, 1, 0, 0, 0, 0, 0, ..., 0

        id2word = generator.id2word

        x_words = [id2word[id] for id in x[0]]
        print(x_words)
        >>> <S> I have a <UNK> </S> <PAD> ... <PAD>
    '''
    def __init__(self, cfg, path, B, T=40, min_count=1, shuffle=True):
        self.path = path
        self.B = B
        self.T = T

        self.vocab = Vocab()
        sentences = load_data(path)
        self.vocab.build_vocab(sentences)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        # self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        with open(path, 'r', encoding='utf-8') as f:
            self.n_data = sum(1 for line in f)
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.training_indices, self.validation_indices = prepare_training_validation_indices(sentences, cfg['train_val_split_percent'])
        self.reset()


    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            None: no input is needed for generator pretraining.
            x: numpy.array, shape = (B, max_length)
            y_true: numpy.array, shape = (B, max_length, V)
                labels with one-hot encoding.
                max_length is the max length of sequence in the batch.
                if length smaller than max_length, the data will be padded.
        '''
        x, y_true = [], []
        start = idx * self.B + 1
        end = (idx + 1) * self.B + 1
        max_length = 0
        for i in range(start, end):
            if self.shuffle:
                idx = self.shuffled_indices[i]
            else:
                idx = i
            sentence = linecache.getline(self.path, idx) # str
            words = sentence.strip().split()  # list of str
            ids = sentence_to_ids(self.vocab, words) # list of ids

            ids_x, ids_y_true = [], []

            ids_x.append(self.vocab.BOS)
            ids_x.extend(ids)
            ids_x.append(self.vocab.EOS) # ex. [BOS, 8, 10, 6, 3, EOS]
            x.append(ids_x)

            ids_y_true.extend(ids)
            ids_y_true.append(self.vocab.EOS) # ex. [8, 10, 6, 3, EOS]
            y_true.append(ids_y_true)

            max_length = max(max_length, len(ids_x))

        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(x):
            x[i] = x[i][:max_length]
        for i, ids in enumerate(y_true):
            y_true[i] = y_true[i][:max_length]

        x = [pad_seq(sen, max_length) for sen in x]
        x = np.array(x, dtype=np.int32)

        y_true = [pad_seq(sen, max_length) for sen in y_true]
        y_true = np.array(y_true, dtype=np.int32)
        y_true = to_categorical(y_true, num_classes=self.V)

        return (x, y_true)

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        x, y_true = self.__getitem__(self.idx)
        self.idx += 1
        return (x, y_true)

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.shuffled_indices = np.arange(self.n_data)
            np.random.shuffle(self.shuffled_indices)

    def on_epoch_end(self):
        self.reset()
        pass


class DiscriminatorGenerator(Sequence):
    '''
    Generate generator pretraining data.
    # Arguments
        path_pos: str, path to true data
        path_neg: str, path to generated data
        B: int, batch size
        T (optional): int or None, default is None.
            if int, T is the max length of sequential data.
        min_count (optional): int, minimum of word frequency for building vocabrary
        shuffle (optional): bool

    # Params
        PAD, BOS, EOS, UNK: int, id
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN: str
        B, min_count: int
        vocab: Vocab
        word2id: Vocab.word2id
        id2word: Vocab.id2word
        raw_vocab: Vocab.raw_vocab
        V: the size of vocab
        n_data: the number of rows of data

    # Examples
        generator = VAESequenceGenerator('./data/train_x.txt', 32)
        X, Y = generator.__getitem__(idx=11)
        print(X[0])
        >>> 8, 10, 6, 3, 2, 0, 0, ..., 0
        print(Y)
        >>> 0, 1, 1, 0, 1, 0, 0, ..., 1

        id2word = generator.id2word

        x_words = [id2word[id] for id in X[0]]
        print(x_words)
        >>> I have a <UNK> </S> <PAD> ... <PAD>
    '''
    def __init__(self, path_pos, path_neg, B, T=40, min_count=1, shuffle=True):
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.B = B
        self.T = T

        self.vocab = Vocab()
        sentences = load_data(path_pos)
        self.vocab.build_vocab(sentences)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        # self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        with open(path_pos, 'r', encoding='utf-8') as f:
            self.n_data_pos = sum(1 for line in f)
        with open(path_neg, 'r', encoding='utf-8') as f:
            self.n_data_neg = sum(1 for line in f)
        self.n_data = self.n_data_pos + self.n_data_neg
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.reset()

    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            None: no input is needed for generator pretraining.
            X: numpy.array, shape = (B, max_length)
            Y: numpy.array, shape = (B, )
                labels indicate whether sentences are true data or generated data.
                if true data, y = 1. Else if generated data, y = 0.
        '''
        X, Y = [], []
        start = idx * self.B + 1
        end = (idx + 1) * self.B + 1
        max_length = 0
        for i in range(start, end):
            idx = self.indicies[i]
            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                sentence = linecache.getline(self.path_pos, idx) # str
            elif is_pos == 0:
                sentence = linecache.getline(self.path_neg, idx) # str

            words = sentence.strip().split()  # list of str
            ids = sentence_to_ids(self.vocab, words) # list of ids

            x = []
            x.extend(ids)
            x.append(self.vocab.EOS) # ex. [8, 10, 6, 3, EOS]
            X.append(x)
            Y.append(is_pos)

            max_length = max(max_length, len(x))

        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(X):
            X[i] = X[i][:max_length]

        X = [pad_seq(sen, max_length) for sen in X]
        X = np.array(X, dtype=np.int32)

        return (X, Y)

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        X, Y = self.__getitem__(self.idx)
        self.idx += 1
        return (X, Y)

    def reset(self):
        self.idx = 0
        pos_indices = np.arange(start=1, stop=self.n_data_pos+1)
        neg_indices = -1 * np.arange(start=1, stop=self.n_data_neg+1)
        self.indicies = np.concatenate([pos_indices, neg_indices])
        if self.shuffle:
            np.random.shuffle(self.indicies)

    def on_epoch_end(self):
        self.reset()
        pass


def load_texts_from_file(file_path, header=True, delim='\n', is_csv=False):
    '''

    :param file_path:
    :param header:
    :param delim:
    :param is_csv:
    :return: a list of word sequences, each sequence is a list of words
    '''
    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        if header:
            f.readline()
        if is_csv:
            texts = []
            reader = csv.reader(f)
            for row in reader:
                texts.append(row[0])
        else:
            texts = [text_to_word_sequence(line.rstrip(delim), filters='') for line in f]

    return texts

def prepare_training_validation_indices(texts, training_percent):
    '''
    Given the entire text set, split training and validation indicies.
    The indices are all combinations of text indices + token indices
    E.g., [0,0] is the first sentence's first word.
    :param texts:
    :param training_percent:
    :return: training_indicies, validation_indicies
    '''
    indices_list = [np.meshgrid(np.array(i), np.arange(
        len(text) + 1)) for i, text in enumerate(texts)]
    indices_list = np.block(indices_list)

    indices_mask = np.random.rand(indices_list.shape[0]) < training_percent
    return indices_list[indices_mask, :], indices_list[~indices_mask, :]

def generate_sequences_from_texts(texts, indices_list,
                                  textgenrnn, context_labels,
                                  batch_size=128):
    is_words = textgenrnn.config['word_level']
    is_single = textgenrnn.config['single_text']
    max_length = textgenrnn.config['max_length']
    meta_token = textgenrnn.META_TOKEN  # Kind of EOS token

    if is_words:
        new_tokenizer = Tokenizer(filters='', char_level=True)
        new_tokenizer.word_index = textgenrnn.vocab
    else:
        new_tokenizer = textgenrnn.tokenizer

    while True:
        np.random.shuffle(indices_list)

        X_batch = []
        Y_batch = []
        context_batch = []
        count_batch = 0

        for row in range(indices_list.shape[0]):
            text_index = indices_list[row, 0]
            end_index = indices_list[row, 1]

            text = texts[text_index]

            if not is_single:
                text = [meta_token] + list(text) + [meta_token]

            if end_index > max_length:
                x = text[end_index - max_length: end_index + 1]
            else:
                x = text[0: end_index + 1]
            y = text[end_index + 1]

            if y in textgenrnn.vocab:
                x = process_sequence([x], textgenrnn, new_tokenizer)
                y = textgenrnn_encode_cat([y], textgenrnn.vocab)

                X_batch.append(x)
                Y_batch.append(y)

                if context_labels is not None:
                    context_batch.append(context_labels[text_index])

                count_batch += 1

                if count_batch % batch_size == 0:
                    X_batch = np.squeeze(np.array(X_batch))
                    Y_batch = np.squeeze(np.array(Y_batch))
                    context_batch = np.squeeze(np.array(context_batch))

                    # print(X_batch.shape)

                    if context_labels is not None:
                        yield ([X_batch, context_batch], [Y_batch, Y_batch])
                    else:
                        yield (X_batch, Y_batch)
                    X_batch = []
                    Y_batch = []
                    context_batch = []
                    count_batch = 0


def process_sequence(X, textgenrnn, new_tokenizer):
    X = new_tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(
        X, maxlen=textgenrnn.config['max_length'])

    return X