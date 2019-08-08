from .constants import config, MODEL, USE_GPU, DATA_ROOT, ENCODERS
from .dataset import build_vocab, build_reader
from .model import BaselineModel, build_embeddings, build_encoder
from .test import calculate_metrics

import torch
import torch.optim as optim

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.manual_seed(config.seed)

    reader = build_reader(MODEL)
    train_ds, test_ds = (reader.read(DATA_ROOT.format(fname)) for fname in ["train.json", "test.json"])
    
    vocab = build_vocab(reader, config.max_vocab_size, MODEL)
    iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    word_embeddings = build_embeddings(MODEL, config)
    encoder =  build_encoder(MODEL, config, word_embeddings)

    model = BaselineModel(
        word_embeddings, 
        encoder, 
        vocab
    )

    if USE_GPU:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        cuda_device=0 if USE_GPU else -1,
        num_epochs=config.epochs,
    )

    metrics = trainer.train()
    print()
    print(metrics)

    ax = calculate_metrics(model, test_ds, vocab, batch_size=32 if MODEL == 'elmo' else 128)
    plt.show()


    
    