from constants import USE_GPU, DATA_ROOT, ENCODERS, config_defaults
from config import Config
from dataset import build_vocab, build_reader
from model import BaselineModel, build_embeddings, build_encoder, build_model
from test import calculate_metrics

import argparse 

import torch
from torch import nn
import torch.optim as optim

from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

import matplotlib.pyplot as plt

import neptune as nt

def build_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model', default='base', type=str)
    parser.add_argument('-i', '--indexer', default='base', type=str)
    parser.add_argument('-f','--features', default='union', type=str)

    parser.add_argument('-e', '--encoder', type=str)
    parser.add_argument('-d', '--embedding_dim', type=int, default=512)

    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('-bsz', '--batch_size', default=8, type=int)
    parser.add_argument('-ep', '--epochs', default=15, type=int)

    parser.add_argument('-t', '--track', action='store_true')#, default=False, type=bool)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = build_argparser()
    
    cfg_dict = {**config_defaults, **vars(args)}
    config = Config(tags=['allennlp', args.features, args.encoder], **cfg_dict)

    config.parameter('encoder', config.encoder)
    config.parameter('features', config.features)

    torch.manual_seed(config.seed)

    reader = build_reader(config)
    
    url = DATA_ROOT[config.features]
    train_ds, test_ds = (reader.read(url.format(fname)) for fname in ["train.json", "test.json"])
    
    model, trainer, vocab = build_model(config, reader, train_ds)

    metrics = trainer.train()
    print()
    print(metrics)

    for k, v in metrics.items():
        try:
            config.log(k, float(v))
        except:
            pass

    img, acc, f1 = calculate_metrics(model, config.features, test_ds, vocab, batch_size=32 if config.encoder == 'bert' else 128)
    if config.track:
        config.log('acc', acc)
        config.log('f1', f1)

        config.exp.send_image('confusion-matrix', img)
        nt.stop()

    plt.show()


    
    