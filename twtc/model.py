from typing import *

from constants import LABEL_COLS, FEATURE_COLS, ELMO_URLS, GPT2_URL, ENCODERS, USE_GPU
from dataset import build_vocab

import torch
from torch import nn
from torch import optim

import numpy as np

from allennlp.models import Model, BasicClassifier, BertForClassification
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from allennlp.modules.token_embedders import Embedding, TokenEmbedder, TokenCharactersEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder, OpenaiTransformerEmbedder, PretrainedBertEmbedder

from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder, CnnHighwayEncoder

class BaselineModel(Model):
    def __init__(self, 
                features: str,
                word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 feature_cols=FEATURE_COLS,
                 out_sz: int=len(LABEL_COLS),
                 ):
        super().__init__(vocab)
        
        hidden_sz = encoder.get_output_dim() + (len(feature_cols) if features == 'union' else 0)
        self.features = features
        
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(hidden_sz, out_sz)
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, 
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor,
                features: np.ndarray = None,
        ) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        feature_vector = torch.tensor(self.encoder(embeddings, mask))
        
        #features = torch.tensor(features)
        
        if self.features == 'union':
            feature_vector = torch.cat([feature_vector, features], dim=1)
        class_logits = self.projection(feature_vector)
        
        output = {"class_logits": class_logits}
        output["loss"] = self.loss(class_logits, label)

        return output


def build_token_embeddings(config):
    embeddings = config.indexer
    if 'base' in embeddings:
        assert config
        
        token_embedding = Embedding(num_embeddings=config.max_vocab_size + 2, embedding_dim=config.embedding_dim, padding_index=0)
        if 'char' in embeddings:
            return TokenCharactersEncoder(token_embedding, BagOfEmbeddingsEncoder(token_embedding.get_output_dim())) 
        
        return token_embedding
    elif embeddings == 'elmo':
        return ElmoTokenEmbedder(**ELMO_URLS)
    elif embeddings == 'gpt2':
        return OpenaiTransformerEmbedder(GPT2_URL)
    elif embeddings == 'bert':
        return PretrainedBertEmbedder(
                pretrained_model="bert-base-uncased",
                top_layer_only=True, # conserve memory
        )

    raise ValueError(f'Invalid embeddings: {embeddings}')

def build_embeddings(config) -> TextFieldEmbedder:
    token_embedding = build_token_embeddings(config)
    return BasicTextFieldEmbedder({"tokens": token_embedding}, allow_unmatched_keys = True)  


def build_encoder(word_embeddings, config):
    EncoderModel = ENCODERS[config.encoder]
    args = [word_embeddings.get_output_dim()]

    if config.model == 'bert':
        return EncoderModel()
    elif config.model == 'highway-cnn':
        args.extend([[(i, config.hidden_sz) for i in range(2, 6)], 2, config.hidden_sz])
    else:
        args.append(config.hidden_sz)

    args = tuple(args)
    encoder = EncoderModel(*args)

    print(f'Using {type(encoder).__name__} as encoder.')
    return encoder


def build_model(config, reader, train_ds, vocab=None):
    if not vocab:
        vocab = build_vocab(config, reader)
    iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    if config.model == 'bert':
        model = BertForClassification(vocab, 'bert-base-uncased', num_labels=2, index='tokens')
    else:
        word_embeddings = build_embeddings(config)
        encoder =  build_encoder(word_embeddings, config)

        if config.features == 'union':
            model = BaselineModel(
                config.features,
                word_embeddings, 
                encoder, 
                vocab,
                feature_cols=reader.feature_cols
            )
        elif config.features == 'text':
            model = BasicClassifier(
            vocab,
            word_embeddings,
            encoder
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

    return model, trainer, vocab

    