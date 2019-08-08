from typing import *

from .constants import TOKEN_INDEXERS, DATA_ROOT, LABEL_COLS
from .config import Config

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



def clean_text(corp, rm_null=True):
    if rm_null:
        corp = corp[~corp.isnull()]
    corp = corp.str.lower().str.encode('ascii', 'ignore').str.decode('ascii')
    corp = corp.str.replace("bernie pleskoff's scouting report", "")
    corp = corp.apply(lambda doc: ''.join([char if char.isalnum() else f' {char} ' if doc else None for char in doc]))
    return corp

class MLBDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int]=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_seq_len = max_seq_len

    @overrides
    def text_to_instance(self, tokens: List[Token], id: str,
                         labels: np.ndarray) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        
        id_field = MetadataField(id)
        fields["id"] = id_field
        
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        # df = pd.read_csv(file_path)
        df = pd.read_json(file_path)
        df = df.reset_index().rename(columns={'Index': 'id'})
        df = df[df.report.str.split(' ').str.len() > 3]
        df['report'] = clean_text(df['report'], rm_null=False)
        
        data = df.apply(lambda row: self.text_to_instance(
                [Token(x) for x in self.tokenizer(row["report"])],
                row["index"], 
            row[LABEL_COLS].values,
            ), axis=1)
        return iter(data)

def tokenizer(x: str, max_seq_len:int):
    return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', 
                                pos_tags=False).split_words(x)[:max_seq_len]]

def build_reader(indexer):
    reader = MLBDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": TOKEN_INDEXERS[indexer]()}
    )
    return reader


def build_vocab(reader=None, max_vocab_size=None, model='base'):
    if model == 'base':
        assert max_vocab_size
        assert reader

        full_ds = reader.read(DATA_ROOT.format("full_dataset.json"))
        return Vocabulary.from_instances(full_ds, max_vocab_size=max_vocab_size)
    return Vocabulary()