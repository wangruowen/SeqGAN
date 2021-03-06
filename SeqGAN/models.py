import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Dropout, Concatenate
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Bidirectional
from keras.layers import Activation
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical
import tensorflow as tf
import pickle
from .utils import *

def new_lstm(rnn_size, layer_num, use_bidirectional=False, return_sequences=True, return_state=False):
    use_cudnnlstm = K.backend() == 'tensorflow' and len(K.tensorflow_backend._get_available_gpus()) > 0
    if use_cudnnlstm:
        from keras.layers import CuDNNLSTM
        if use_bidirectional:
            return Bidirectional(CuDNNLSTM(rnn_size,
                                           return_sequences=return_sequences, return_state=return_state),
                                 name='rnn_{}'.format(layer_num))

        return CuDNNLSTM(rnn_size,
                         return_sequences=return_sequences, return_state=return_state,
                         name='rnn_{}'.format(layer_num))
    else:
        if use_bidirectional:
            return Bidirectional(LSTM(rnn_size,
                                      return_sequences=return_sequences, return_state=return_state,
                                      recurrent_activation='sigmoid'),
                                 name='rnn_{}'.format(layer_num))

        return LSTM(rnn_size,
                    return_sequences=return_sequences, return_state=return_state,
                    recurrent_activation='sigmoid',
                    name='rnn_{}'.format(layer_num))


def GeneratorPretraining(cfg, vocab):
    '''
    Model for Generator pretraining. This model's weights should be shared with
        Generator.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
    # Returns:
        generator_pretraining: keras Model
            input: word ids, shape = (B, T)
            output: word probability, shape = (B, T, V)
    '''
    # in comment, B means batch size, T means lengths of time steps.
    input = Input(shape=(cfg['max_length'],), dtype='int32', name='Input')  # (B(not included), T)
    embedded = Embedding(vocab.num_classes, cfg['gen_embed'], input_length=cfg['max_length'],
                         name='Embedding')(input)  # (B(not included), T, E) mask_zero=True,

    prev_layer = embedded
    for i in range(cfg['rnn_layers']):
        if i < cfg['rnn_layers'] - 1:
            return_seq = True  # (B, T, H)
        else:
            return_seq = False  # Last LSTM return only last output (B, H)
        prev_layer = new_lstm(cfg['gen_hidden'], i + 1, return_sequences=return_seq)(prev_layer)

    out = Dense(vocab.num_classes, activation='softmax', name='DenseSoftmax')(prev_layer)
    generator_pretraining = Model(inputs=input, outputs=out)
    return generator_pretraining

class Generator():
    'Create Generator, which generate a next word.'
    def __init__(self, sess, cfg, vocab):
        '''
        # Arguments:
            B: int, Batch size
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
        '''
        self.sess = sess
        self.cfg = cfg
        self.vocab = vocab
        self.B = cfg['batch_size']
        self.T = cfg['max_length']
        self.V = vocab.num_classes
        self.E = cfg['gen_embed']
        self.H = cfg['gen_hidden']
        self.lr = cfg['gen_lr']
        self._build_gragh()
        self.reset_rnn_state()

    def _build_gragh(self):
        state_in = tf.placeholder(tf.float32, shape=(None, 1), name='state_in')  # (B, 1)
        self.init_hs, self.init_cs = [], []
        self.curr_hs, self.curr_cs = [None] * self.cfg['rnn_layers'], [None] * self.cfg['rnn_layers']
        self.next_hs, self.next_cs = [], []
        for i in range(self.cfg['rnn_layers']):
            self.init_hs.append(tf.placeholder(tf.float32, shape=(None, self.H), name='rnn_%d_init_h' % (i + 1)))
            self.init_cs.append(tf.placeholder(tf.float32, shape=(None, self.H), name='rnn_%d_init_c' % (i + 1)))
        action = tf.placeholder(tf.float32, shape=(None, self.V), name='action')  # onehot (B, V)
        reward = tf.placeholder(tf.float32, shape=(None, ), name='reward')  # (B, )

        self.layers = []

        embedding = Embedding(self.V, self.E, name='Embedding')  # mask_zero=True,
        embedded = embedding(state_in)  # (B, 1, E)
        self.layers.append(embedding)

        out = embedded  # (B, 1, E)
        # Since the state_in.shape = (B, 1), this is feeding one step at a time into the model, not the entire sequence
        for i in range(self.cfg['rnn_layers']):
            # print("i = ", i)
            if i < self.cfg['rnn_layers'] - 1:
                return_seq = True  # out.shape = (B, 1, H)
            else:
                return_seq = False  # Last LSTM return only last output (B, H)
            cur_lstm = new_lstm(self.H, i + 1, return_sequences=return_seq, return_state=True)
            out, next_h, next_c = cur_lstm(out, initial_state=[self.init_hs[i], self.init_cs[i]])
            self.next_hs.append(next_h)
            self.next_cs.append(next_c)
            self.layers.append(cur_lstm)

        # For each time step, we output to a Dense with softmax. But actually, we should only need to do this at time T.
        dense = Dense(self.V, activation='softmax', name='DenseSoftmax')
        prob = dense(out)    # (B, V)
        self.layers.append(dense)

        # ATTENTION! This loss function causes mode collapse problem!
        # We need to come up a new loss function to train
        log_prob = tf.log(tf.reduce_mean(prob * action, axis=-1))  # (B, )
        loss = - tf.reduce_mean(log_prob * reward)  # sum up the entire batch
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        minimize = optimizer.minimize(loss)

        self.state_in = state_in
        self.action = action
        self.reward = reward
        self.prob = prob
        self.minimize = minimize
        self.loss = loss

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def reset_rnn_state(self, batch_size=None):
        if batch_size is None:
            batch_size = self.B
        for i in range(self.cfg['rnn_layers']):
            self.curr_hs[i] = np.zeros([batch_size, self.H])
            self.curr_cs[i] = np.zeros([batch_size, self.H])

    def warm_up_rnn_state(self, batch_size=None):
        """
        Warm up the RNN
        Note, this is important, since this model's weight is copied from
        the pre-trained generator, which assumes an input of T time steps.
        We need to do the same thing here to warm up the RNN (basically, the hidden and cell state)
        :param batch_size:
        :return:
        """
        if batch_size is None:
            batch_size = self.B
        pad = np.zeros((batch_size, 1))
        for _ in range(self.T - 1):
            self.predict(pad)

    def set_rnn_state(self, hs, cs):
        '''
        # Arguments:
            h: np.array, shape = (B,H)
            c: np.array, shape = (B,H)
        '''
        for i in range(self.cfg['rnn_layers']):
            self.curr_hs[i] = hs[i]
            self.curr_cs[i] = cs[i]

    def get_rnn_state(self):
        return self.curr_hs, self.curr_cs

    def predict(self, state, stateful=True):
        '''
        Predict next action(word) probability
        # Arguments:
            state: np.array, previous word ids, shape = (B, 1)
        # Optional Arguments:
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return prob.
                else, return prob, next_h, next_c without updating states.
        # Returns:
            prob: np.array, shape=(B, V)
        '''
        # state = state.reshape(-1, 1)
        feed_dict = {self.state_in: state}
        for i in range(self.cfg['rnn_layers']):
            # Use current h and c to feed into initial_state
            feed_dict[self.init_hs[i]] = self.curr_hs[i]
            feed_dict[self.init_cs[i]] = self.curr_cs[i]

        result_dict = self.sess.run({'prob': self.prob,
                                     'next_hs': self.next_hs,
                                     'next_cs': self.next_cs}, feed_dict=feed_dict)
        prob = result_dict['prob']
        next_hs = result_dict['next_hs']
        next_cs = result_dict['next_cs']

        if stateful:
            self.set_rnn_state(next_hs, next_cs)
            return prob
        else:
            return prob, next_hs, next_cs

    def update(self, state, action, reward, hs=None, cs=None, stateful=True):
        '''
        Update weights by Policy Gradient.
        # Arguments:
            state: np.array, Environment state, shape = (B, 1) or (B, t)
                if shape is (B, t), state[:, -1] will be used.
            action: np.array, Agent action, shape = (B, )
                In training, action will be converted to onehot vector.
                (Onehot shape will be (B, V))
            reward: np.array, reward by Environment, shape = (B, )

        # Optional Arguments:
            h: np.array, shape = (B, H), default is None.
                if None, h will be Generator.h
            c: np.array, shape = (B, H), default is None.
                if None, c will be Generator.c
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return loss.
                else, return loss, next_h, next_c without updating states.

        # Returns:
            loss: np.array, shape = (B, )
            next_h: (if stateful is True)
            next_c: (if stateful is True)
        '''
        if hs is None:
            hs = self.curr_hs
        if cs is None:
            cs = self.curr_cs
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1)
        feed_dict = {
            self.state_in: state,
            self.action: to_categorical(action, self.V),
            self.reward: reward
        }
        for i in range(self.cfg['rnn_layers']):
            # Use current h and c to feed into initial_state
            feed_dict[self.init_hs[i]] = hs[i]
            feed_dict[self.init_cs[i]] = cs[i]

        result_dict = self.sess.run(
            {'minimize': self.minimize, 'loss': self.loss,
             'next_hs': self.next_hs, 'next_cs': self.next_cs},
            feed_dict=feed_dict)

        loss = result_dict['loss']
        next_hs = result_dict['next_hs']
        next_cs = result_dict['next_cs']

        if stateful:
            self.set_rnn_state(next_hs, next_cs)
            return loss
        else:
            return loss, next_hs, next_cs

    def sampling_word(self, prob):
        '''
        # Arguments:
            prob: numpy array, dtype=float, shape = (B, V),
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        '''
        return sample_one_word(prob)

    def sampling_sentence(self, T):
        '''
        # Arguments:
            T: int, max time steps
        # Optional Arguments:
            BOS: int, id for Begin Of Sentence
        # Returns:
            actions: numpy array, dtype=int, shape = (B, T)
        '''
        # Because each generated sentence may have different length, it is easy to
        # use a loop than batch
        sentences = []
        for _ in range(self.B):
            self.reset_rnn_state(1)  # Batchsize = 1

            # Warm up the RNN
            # Note, this is important, since this model's weight is copied from
            # the pre-trained generator, which assumes an input of T time steps.
            # We need to do the same thing here to warm up the RNN (basically, the hidden and cell state)
            pad = np.array([self.vocab.PAD]).reshape(1, 1)
            for _ in range(self.T - 1):
                self.predict(pad)

            action = np.array([self.vocab.BOS]).reshape(1, 1)
            actions = action
            for _ in range(T):
                prob = self.predict(action)  # (1, V)
                action = sample_one_word(prob)  # (1, 1)
                if action[0, 0] == self.vocab.EOS:
                    break
                # action = action.reshape(1, 1)
                actions = np.concatenate([actions, action], axis=-1)

            sentences.append(actions[0, 1:])  # Remove BOS

        return sentences  # list of 1D arrays

    def generate_samples(self, T, vocab, num, output_file):
        '''
        Generate sample sentences to output file
        # Arguments:
            T: int, max time steps
            g_data: SeqGAN.utils.GeneratorPretrainingGenerator
            num: int, number of sentences
            output_file: str, path
        '''
        sentences=[]
        for _ in range(num // self.B + 1):
            actions_list = self.sampling_sentence(T)
            # actions_list = actions.tolist()
            for sentence_id in actions_list:
                sentence = []
                for action in sentence_id:
                    if action == vocab.EOS:
                        break
                    sentence.append(vocab.id2word[action])
                sentences.append(sentence)
        output_str = ''
        for i in range(num):
            output_str += ' '.join(sentences[i]) + '\n'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)

    # TODO replace pickle and use keras's save/load api
    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

def Discriminator(cfg, vocab, use_highway=True):
    '''
    Disciriminator model.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1)
    '''
    input = Input(shape=(cfg['max_length'],), dtype='int32', name='Input')   # (B, T)
    out = Embedding(vocab.num_classes, cfg['dis_embed'], name='Embedding')(input)  # (B, T, E) mask_zero=True,

    for i in range(cfg['rnn_layers']):
        if i < cfg['rnn_layers'] - 1:
            return_seq = True  # (B, T, H)
        else:
            return_seq = False  # Last LSTM returns only last output (B, H)
        out = new_lstm(cfg['dis_hidden'], i + 1, return_sequences=return_seq)(out)

    if use_highway:
        out = Highway(out, num_layers=1)
    out = Dropout(cfg['dis_dropout'], name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out)
    return discriminator

def DiscriminatorConv(V, E, filter_sizes, num_filters, dropout):
    '''
    Another Discriminator model, currently unused because keras don't support
    masking for Conv1D and it does huge influence on training.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        filter_sizes: list of int, list of each Conv1D filter sizes
        num_filters: list of int, list of each Conv1D num of filters
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1)
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, name='Embedding')(input)  # (B, T, E)
    out = VariousConv1D(out, filter_sizes, num_filters)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out)
    return discriminator

def VariousConv1D(x, filter_sizes, num_filters, name_prefix=''):
    '''
    Layer wrapper function for various filter sizes Conv1Ds
    # Arguments:
        x: tensor, shape = (B, T, E)
        filter_sizes: list of int, list of each Conv1D filter sizes
        num_filters: list of int, list of each Conv1D num of filters
        name_prefix: str, layer name prefix
    # Returns:
        out: tensor, shape = (B, sum(num_filters))
    '''
    conv_outputs = []
    for filter_size, n_filter in zip(filter_sizes, num_filters):
        conv_name = '{}VariousConv1D/Conv1D/filter_size_{}'.format(name_prefix, filter_size)
        pooling_name = '{}VariousConv1D/MaxPooling/filter_size_{}'.format(name_prefix, filter_size)
        conv_out = Conv1D(n_filter, filter_size, name=conv_name)(x)   # (B, time_steps, n_filter)
        conv_out = GlobalMaxPooling1D(name=pooling_name)(conv_out) # (B, n_filter)
        conv_outputs.append(conv_out)
    concatenate_name = '{}VariousConv1D/Concatenate'.format(name_prefix)
    out = Concatenate(name=concatenate_name)(conv_outputs)
    return out

def Highway(x, num_layers=1, activation='relu', name_prefix=''):
    '''
    Layer wrapper function for Highway network
    # Arguments:
        x: tensor, shape = (B, input_size)
    # Optional Arguments:
        num_layers: int, dafault is 1, the number of Highway network layers
        activation: keras activation, default is 'relu'
        name_prefix: str, default is '', layer name prefix
    # Returns:
        out: tensor, shape = (B, input_size)
    '''
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x
