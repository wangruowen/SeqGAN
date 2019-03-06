from .models import GeneratorPretraining, Discriminator, Generator
from .utils import *
from .rl import Agent, Environment
from keras.optimizers import Adam
import os
import numpy as np
import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)

class Trainer(object):
    '''
    Manage training
    '''
    def __init__(self, cfg, training_set_path, init_eps=0.1):
        self.cfg = cfg
        self.B, self.T = cfg['batch_size'], cfg['max_length']
        self.g_E, self.g_H = cfg['gen_embed'], cfg['gen_hidden']
        self.d_E, self.d_H = cfg['dis_embed'], cfg['dis_hidden']
        self.d_dropout = cfg['dis_dropout']
        self.generate_samples = cfg['gen_samples']
        self.g_lr, self.d_lr = cfg['gen_lr'], cfg['dis_lr']
        self.eps = init_eps
        self.init_eps = init_eps
        self.top = os.getcwd()
        self.path_pos = training_set_path
        self.path_neg = os.path.join(self.top, 'data', 'save', 'generated_sentences.txt')

        self.texts = load_texts(self.path_pos)
        self.texts_lines = len(self.texts)
        self.vocab = create_vocabulary(cfg, self.texts)
        self.pretrain_data_generator = generate_pretrain_batch(cfg, self.texts, self.vocab)

        self.V = self.vocab.num_classes
        self.agent = Agent(sess, cfg, self.vocab)
        self.g_beta = Agent(sess, cfg, self.vocab)
        self.discriminator = Discriminator(cfg, self.vocab, use_highway=False)
        self.env = Environment(cfg, self.discriminator, self.vocab, self.g_beta, n_sample=cfg['mcts_sample'])

        self.generator_pre = GeneratorPretraining(cfg, self.vocab)

    def pre_train(self, g_epochs=3, d_epochs=1, g_weight_file=None, d_weight_file=None,
        g_lr=1e-3, d_lr=1e-3):
        self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_weight_file, lr=g_lr)
        self.pre_train_discriminator(d_epochs=d_epochs, d_pre_path=d_weight_file, lr=d_lr)

    def pre_train_generator(self, g_epochs=3, g_pre_path=None, lr=1e-3):
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.hdf5')
        else:
            self.g_pre_path = g_pre_path

        g_adam = Adam(lr)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')
        print('Generator pre-training')
        self.generator_pre.summary()

        self.generator_pre.fit_generator(
            self.pretrain_data_generator,
            steps_per_epoch=max(int(np.floor(self.texts_lines / self.B)), 1),
            epochs=g_epochs,
            verbose=1)
        self.generator_pre.save_weights(self.g_pre_path)
        self.reflect_pre_train()

    def pre_train_discriminator(self, d_epochs=1, d_pre_path=None, lr=1e-3):
        if d_pre_path is None:
            self.d_pre_path = os.path.join(self.top, 'data', 'save', 'discriminator_pre.hdf5')
        else:
            self.d_pre_path = d_pre_path

        print('Start Generating sentences')
        self.agent.generator.generate_samples(self.T, self.vocab,
            self.generate_samples, self.path_neg)

        self.neg_texts = load_texts(self.path_neg)
        self.neg_texts_lines = len(self.neg_texts)
        self.train_data_generator = generate_train_batch(self.cfg, self.texts, self.neg_texts, self.vocab)

        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        print('Discriminator pre-training')

        self.discriminator.fit_generator(
            self.train_data_generator,
            steps_per_epoch=max(int(np.floor(self.neg_texts_lines / self.B)), 1),
            epochs=d_epochs,
            verbose=1)
        self.discriminator.save(self.d_pre_path)

    def load_pre_train_g(self, g_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()

    def load_pre_train_d(self, d_pre_path):
        self.discriminator.load_weights(d_pre_path)

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.load_pre_train_g(g_pre_path)
        self.load_pre_train_d(d_pre_path)

    def reflect_pre_train(self):
        i = 0
        for layer in self.generator_pre.layers:
            if len(layer.get_weights()) != 0:
                w = layer.get_weights()
                self.agent.generator.layers[i].set_weights(w)
                self.g_beta.generator.layers[i].set_weights(w)
                i += 1

    def train(self, steps=10, g_steps=1, d_steps=1, d_epochs=1,
        g_weights_path='data/save/generator.pkl',
        d_weights_path='data/save/discriminator.hdf5',
        verbose=True,
        head=1):
        d_adam = Adam(self.d_lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.eps = self.init_eps
        for step in range(steps):
            # Generator training
            for _ in range(g_steps):
                rewards = np.zeros([self.B, self.T])
                self.agent.reset()
                self.env.reset()
                for t in range(self.T):
                    state = self.env.get_state()
                    action = self.agent.act(state, epsilon=0.0)
                    next_state, reward, is_episode_end, info = self.env.step(action)
                    self.agent.generator.update(state, action, reward)
                    rewards[:, t] = reward.reshape([self.B, ])
                    if is_episode_end:
                        if verbose:
                            print('Reward: {:.3f}, Episode end'.format(np.average(rewards)))
                            self.env.render(head=head)
                        break
            # Discriminator training
            for _ in range(d_steps):
                self.agent.generator.generate_samples(
                    self.T,
                    self.g_data,
                    self.generate_samples,
                    self.path_neg)
                self.d_data = DiscriminatorGenerator(
                    path_pos=self.path_pos,
                    path_neg=self.path_neg,
                    B=self.B,
                    shuffle=True)
                self.discriminator.fit_generator(
                    self.d_data,
                    steps_per_epoch=None,
                    epochs=d_epochs)

            # Update env.g_beta to agent
            self.agent.save(g_weights_path)
            self.g_beta.load(g_weights_path)

            self.discriminator.save(d_weights_path)
            self.eps = max(self.eps*(1- float(step) / steps * 4), 1e-4)

    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(d_path)

    def test(self):
        x, y = self.d_data.next()
        pred = self.discriminator.predict(x)
        for i in range(self.B):
            txt = [self.g_data.id2word[id] for id in x[i].tolist()]
            label = y[i]
            if label == 0:
                print('{}, {:.3f}: {}'.format(label, pred[i,0], ' '.join(txt)))
