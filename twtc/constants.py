from typing import *

from config import Config

import torch
from torch import nn

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder, CnnHighwayEncoder

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import OpenaiTransformerBytePairIndexer

USE_GPU = torch.cuda.is_available()
USE_NT = True

MODEL = 'base'
ENCODER = 'cnn'

config = Config(
    USE_NT,
    tags = [ENCODER, MODEL],
    testing=False,
    seed=1,
    batch_size=64,
    lr=0.01 if MODEL == 'elmo' else 3e-4,
    epochs=10 if MODEL == 'elmo' else 15,
    hidden_sz=256 if MODEL == 'elmo' else 512,
    max_seq_len=512 if MODEL == 'elmo' else None,
    max_vocab_size=100000,
    embedding_dim=300
)

DATA_ROOT = 'https://github.com/jacobdanovitch/Trouble-With-The-Curve/blob/master/data/scouting_classification/{}?raw=true'

LABEL_COLS = ['label']

TOKEN_INDEXERS = {
    'base': SingleIdTokenIndexer,
    'base-char': lambda: TokenCharactersIndexer(min_padding_length=5),
    'elmo': ELMoTokenCharactersIndexer,
    'gpt2': OpenaiTransformerBytePairIndexer
}

ELMO_URLS = dict(
    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json',
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
)

GPT2_URL = 'https://gist.githubusercontent.com/joelgrus/885ef14323ff3d6ecefb84fa59c48039/raw/ec3ef38729e45c6ab2c56dbb30cfc734ae60a1aa/ner-openai.jsonnet'


ENCODERS = {
    'lstm': lambda *args: PytorchSeq2VecWrapper(nn.LSTM(*args, bidirectional=True, batch_first=True)),
    'cnn': CnnEncoder,
    'highway-cnn': CnnHighwayEncoder,
    'boe': BagOfEmbeddingsEncoder
}