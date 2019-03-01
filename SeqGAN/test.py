from utils import generate_batch_from_texts

cfg = {'max_length': 10, 'batch_size': 5, 'max_words': 10000}
for X_batch, Y_batch in generate_batch_from_texts(cfg, '../trump'):
    print(X_batch)
    print(Y_batch)