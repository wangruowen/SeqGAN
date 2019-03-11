from .models import Generator, GeneratorPretraining, Discriminator
from .utils import DiscriminatorGenerator
import keras.backend as K
import numpy as np

class Agent(object):
    '''
    On each step, Agent act on state.
    Then Environment return next state, reward, and so on.
    '''
    def __init__(self, sess, cfg, vocab):
        '''
        # Arguments:
            sess: tf.Session
            B: int, batch_size
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
        '''
        self.sess = sess
        self.num_actions = vocab.num_classes
        self.B = cfg['batch_size']
        self.V = vocab.num_classes
        self.E = cfg['gen_embed']
        self.H = cfg['gen_hidden']
        self.lr = cfg['gen_lr']
        self.eps = 0.1
        self.generator = Generator(sess, cfg, vocab)

    def act(self, state, epsilon=0, deterministic=False):
        '''
        # Arguments:
            state: numpy array, dtype=int, shape = (B, t)
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        '''
        word = state[:, -1].reshape([-1, 1])
        return self._act_on_word(word, epsilon=epsilon, deterministic=deterministic)

    def _act_on_word(self, word, epsilon=0, deterministic=False, PAD=0, EOS=2):
        '''
        # Arguments:
            word: numpy array, dtype=int, shape = (B, 1),
                word indicates current word.
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        '''
        # action = None
        is_PAD = word == PAD
        is_EOS = word == EOS
        is_end = is_PAD.astype(np.int) + is_EOS.astype(np.int)  # Either is_PAD or is_EOS, indicates the end
        is_end = 1 - is_end  # if it is end, is_end == 0
        is_end = is_end.reshape([self.B, 1])
        if np.random.rand() <= epsilon:
            action = np.random.randint(low=0, high=self.num_actions, size=(self.B, 1))
        elif not deterministic:
            probs = self.generator.predict(word)
            action = self.generator.sampling_word(probs).reshape([self.B, 1])
        else:
            probs = self.generator.predict(word) # (B, T)
            action = np.argmax(probs, axis=-1).reshape([self.B, 1])
        return action * is_end  # if it is end, return 0.0

    def reset(self):
        self.generator.reset_rnn_state()

    def save(self, path):
        self.generator.save(path)

    def load(self, path):
        self.generator.load(path)


class Environment(object):
    '''
    On each step, Agent act on state.
    Then Environment return next state, reward, and so on.
    '''
    def __init__(self, cfg, discriminator, vocab, g_beta, n_sample=16):
        '''
        Environment class for Reinforced Learning
        # Arguments:
            discriminator: keras model
            data_generator: SeqGAN.models.GeneratorPretrainingGenerator
            g_beta: SeqGAN.rl.Agent, copy of Agent
                params of g_beta.generator should be updated with those of original
                generator on regular occasions.
        # Optional Arguments
            n_sample: int, default is 16, the number of Monte Calro search sample
        '''
        self.vocab = vocab
        self.B = cfg['batch_size']
        self.T = cfg['max_length']
        self.n_sample = n_sample
        self.BOS = vocab.BOS
        self.discriminator = discriminator
        self.g_beta = g_beta
        self.reset()

    def get_state(self):
        # if self.t == 1:
        #     return self._state
        # else:
        #     return self._state[:, 1:]   # Exclude BOS
        return self._state

    def reset(self):
        self.t = 0  # Initially, t = 0 with BOS, Eventually, t = self.T
        self._state = np.zeros([self.B, 1], dtype=np.int32)
        self._state[:, 0] = self.BOS
        self.g_beta.reset()

    def step(self, action):
        '''
        Step t -> t + 1 and returns a result of the Agent action.
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1),
                state is Y_0:t-1, and action is y_t
        # Returns:
            next_state: numpy array, dtype=int, shape = (B, t)
            reward: numpy array, dtype=float, shape = (B, 1)
            is_episode_end: bool
            info: dict
        '''
        reward = self.Q(action, self.n_sample)
        # print("reward: ", reward)
        self.t = self.t + 1
        is_episode_end = self.t == self.T

        self._append_state(action)
        next_state = self.get_state()
        info = None

        return [next_state, reward, is_episode_end, info]

    def render(self, head=1):
        for i in range(head):
            ids = self.get_state()[i]
            words = [self.vocab.id2word[id] for id in ids.tolist()]
            # print(ids.tolist())
            print(' '.join(words))
        print('-' * 80)


    def Q(self, action, n_sample=16):
        '''
        State-Action value function using Rollout policy
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1)

        # Optional Arguments:
            n_sample: int, default is 16, number of samples for Monte Calro Search

        # Returns:
            reward: numpy array, dtype=float, shape = (B, ), State-Action value

        # Requires:
            t, T: used to define time range.
            state: determined texts, Y[0:t-1], used for Rollout.
            action: next words, y[t], used for sentence Y[0:t].
            g_beta: Rollout policy.
        '''
        hs, cs = self.g_beta.generator.get_rnn_state()
        reward = np.zeros([self.B, 1])
        Y_base = self._state
        # print("Call Q at self.t = ", self.t)
        # print("Y_base.shape: ", Y_base.shape)

        # If the input length is already T, then the predict D_phi(Y_1:T) is the reward.
        # That's the second part in the Equation (4) of the original SeqGAN paper.
        if self.t >= self.T:
            print("self.t already >= self.T. Let's directly predict")
            Y = self._append_state(action, state=Y_base)
            return self.discriminator.predict(Y[:, 1:])  # Exclude BOS

        # Monte Carlo Rollout
        for idx_sample in range(n_sample):
            # print("n_sample: ", idx_sample)
            Y = Y_base
            self.g_beta.generator.set_rnn_state(hs, cs)
            # y_t = self.g_beta.act(Y, epsilon=self.g_beta.eps)
            # Y = self._append_state(y_t, state=Y)
            for _ in range(self.t + 1, self.T + 1):  # From t+1 to T
                # print("Rollout: ", tau)
                y_t = self.g_beta.act(Y, epsilon=self.g_beta.eps)
                Y = self._append_state(y_t, state=Y)
                # print("Y.shape: ", Y.shape)
            reward += self.discriminator.predict(Y[:, 1:])

        return reward / n_sample


    def _append_state(self, word, state=None):
        '''
        # Arguments:
            word: numpy array, dtype=int, shape = (B, 1)
        '''
        word = word.reshape(-1, 1)
        if state is None:
            self._state = np.concatenate([self._state, word], axis=-1)
        else:
            return np.concatenate([state, word], axis= -1)
