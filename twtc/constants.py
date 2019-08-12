from typing import *
from overrides import overrides

from config import Config

import torch
from torch import nn

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder, CnnHighwayEncoder, BertPooler

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, PretrainedBertIndexer, OpenaiTransformerBytePairIndexer, ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB as NB


USE_GPU = torch.cuda.is_available()


config_defaults = dict(
    seed=1,
    batch_size=8,#64,
    lr=1e-4,#3e-4,
    epochs=15,
    hidden_sz=512,
    max_seq_len=None,
    max_vocab_size=100000,
    embedding_dim=512
)

DATA_ROOT = {
    'text': 'https://github.com/jacobdanovitch/Trouble-With-The-Curve/blob/master/data/scouting_classification/{}?raw=true',
    'union': 'https://github.com/jacobdanovitch/Trouble-With-The-Curve/blob/master/data/profile_classification/profile_{}?raw=true'
}


LABEL_COLS = ['label']
FEATURE_COLS = ['rank', 'drafted', 'eta', 'Arm', 'Changeup', 'Control', 'Curveball', 'Cutter', 'Fastball', 'Field', 'Hit', 'Overall', 'Power', 'Run', 'Slider', 'Splitter']

TOKEN_INDEXERS = {
    'base': SingleIdTokenIndexer,
    'base-char': lambda: TokenCharactersIndexer(min_padding_length=5),
    'elmo': ELMoTokenCharactersIndexer,
    'gpt2': OpenaiTransformerBytePairIndexer,
    'bert': lambda: PretrainedBertIndexer(
        pretrained_model="bert-base-uncased",
        do_lowercase=True,
        truncate_long_sequences=False
    )
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
    'boe': BagOfEmbeddingsEncoder,
    'bert': lambda: BertPooler(pretrained_model="bert-base-uncased")
}


CLFS = {
    'lr': LogisticRegression(solver='lbfgs', C=1e-2), # good
    'svm': LinearSVC(), # good
    'nb': NB(), 
    'knn': KNeighborsClassifier(10),
    'rf': RandomForestClassifier(n_estimators=100), # , max_features=1
    'nn': MLPClassifier(alpha=1e-2, max_iter=1000), # good
    'ada': AdaBoostClassifier(),
    'sgd': SGDClassifier(), # good
    'gb': GradientBoostingClassifier(), # good 
}

voters = ['lr', 'nn', 'sgd', 'svm']
CLFS['vote'] = VotingClassifier([(v, CLFS[v]) for v in voters])