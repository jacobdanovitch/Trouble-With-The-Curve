from typing import *

from constants import TOKEN_INDEXERS, DATA_ROOT, LABEL_COLS, FEATURE_COLS
from config import Config

from overrides import overrides

import pandas as pd
import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token


from data_utils import clean_text

class MLBReportReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int]=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_seq_len = max_seq_len
        
        #self.cache_data(cache_directory='.data_cache')

    @overrides
    def text_to_instance(self, 
            tokens: List[Token], 
            #id: str,
            labels: np.ndarray
        ) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        #token_type_field = ArrayField(array=np.array([0]*len(tokens)))

        fields = {"tokens": sentence_field} # 'tokens-type-ids': token_type_field}}
        

        #id_field = MetadataField(id)
        #fields["id"] = id_field
        
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_json(file_path)
        df = df.reset_index().rename(columns={'Index': 'id'})
        df = df[df.report.str.split(' ').str.len() > 3]
        df['report'] = clean_text(df['report'], rm_null=False)

        data = df.apply(lambda row: self.text_to_instance(
            [Token(x) for x in self.tokenizer(row["report"], self.max_seq_len)],
            #row["index"],
            row[LABEL_COLS].values,
            ), 
            axis=1)
        return iter(data)

class MLBProfileReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int]=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_seq_len = max_seq_len

        #self.cache_data(cache_directory='.data_cache')

    @overrides
    def text_to_instance(self, 
            tokens: List[Token], 
            features: np.ndarray,
            #id: str,
            labels: np.ndarray
        ) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        #sentence_field.index(Vocabulary())
        #print(vars(sentence_field))
        fields = {"tokens": sentence_field} # ._indexed_tokens

        feature_field = ArrayField(array=features)
        fields['features'] = feature_field
        
        #id_field = MetadataField(id)
        #fields["id"] = id_field
        
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_json(file_path)
        df = df.reset_index().rename(columns={'Index': 'id'})
        df = df[df.report.str.split(' ').str.len() > 3]
        # df['report'] = clean_text(df['report'], rm_null=False)
        df.loc[:, FEATURE_COLS] = (df.loc[:, FEATURE_COLS] - df.loc[:, FEATURE_COLS].mean()) / df.loc[:, FEATURE_COLS].std()
        df = df.fillna(0)

        data = df.apply(lambda row: self.text_to_instance(
            [Token(x) for x in self.tokenizer(row["report"], self.max_seq_len)],
            row[FEATURE_COLS].values,
            #row["index"],
            row[LABEL_COLS].values,
            ), 
            axis=1)
        return iter(data)

def tokenizer(x: str, max_seq_len:int):
    return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', 
                                pos_tags=False).split_words(x)[:max_seq_len]]

def build_reader(config):
    DataReaderClass = MLBProfileReader if config.features == 'union' else MLBReportReader
    reader = DataReaderClass(
        tokenizer=tokenizer,
        token_indexers={"tokens": TOKEN_INDEXERS[config.indexer]()}
    )

    return reader


def build_vocab(config, reader=None):
    if config.indexer == 'base':
        assert reader

        full_ds = reader.read(DATA_ROOT[config.features].format("full_dataset.json"))
        return Vocabulary.from_instances(full_ds, max_vocab_size=config.max_vocab_size)
    return Vocabulary()