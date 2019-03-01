from .train import Trainer

config ={
    'batch_size': 64,
    'max_length': 20,
    'max_words': 10000,
    'gen_embed': 100,
    'dis_embed': 100,
    'gen_hidden': 128,
    'dis_hidden': 128,
    'dis_dropout': 0.2,
    'gen_lr': 1e-3,
    'dis_lr': 1e-3,
    'mcts_sample': 16,
    'gen_samples': 10000,

    'rnn_layers': 3,
    'rnn_bidirectional': False,
}

dataset_path = ""

trainer = Trainer(config, dataset_path)