from typing import *

from .constants import LABEL_COLS, ELMO_URLS, GPT2_URL, ENCODERS

import torch
from torch import nn

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding, TokenEmbedder, TokenCharactersEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder, OpenaiTransformerEmbedder

from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder, CnnHighwayEncoder

class BaselineModel(Model):
    def __init__(self, 
                word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 out_sz: int=len(LABEL_COLS),
                 ):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)
        
        output = {"class_logits": class_logits}
        output["loss"] = self.loss(class_logits, label)

        return output


def build_token_embeddings(model, config):
    if 'base' in model:
        assert config
        
        token_embedding = Embedding(num_embeddings=config.max_vocab_size + 2, embedding_dim=config.embedding_dim, padding_index=0)
        if 'char' in model:
            return TokenCharactersEncoder(token_embedding, BagOfEmbeddingsEncoder(token_embedding.get_output_dim())) 
        
        return token_embedding
    elif model == 'elmo':
        return ElmoTokenEmbedder(*ELMO_URLS)
    elif model == 'gpt2':
        return OpenaiTransformerEmbedder(GPT2_URL)

    raise ValueError(f'Invalid model: {model}')

def build_embeddings(model='base', config=None) -> TextFieldEmbedder:
    token_embedding = build_token_embeddings(model, config)
    return BasicTextFieldEmbedder({"tokens": token_embedding})  


def build_encoder(embeddings, config, model='base'):
    args = [embeddings.get_output_dim()]

    if model == 'highway-cnn':
        args.extend([[(i, config.hidden_sz) for i in range(2, 6)], 2, config.hidden_sz])
    else:
        args.append(config.hidden_sz)

    args = tuple(args)
    return ENCODERS[model](*args)