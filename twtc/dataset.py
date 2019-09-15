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

import torch


from data_utils import clean_text

@DatasetReader.register("mlb")
class MLBDataReader(DatasetReader):
    def __init__(self, 
                 feature_cols = None,
                 standardize = False,
                 tokenizer: Callable[[str], List[str]]=None,
                 indexer: Optional[str]='base',
                 #token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = None
                ) -> None:
        super().__init__(lazy=False)
        
        self.tokenizer = tokenizer or self.spacy_tokenizer
        self.token_indexers = {"tokens": TOKEN_INDEXERS[indexer]()} # config.indexer
        self.max_seq_len = max_seq_len
        
        self.feature_cols = feature_cols
        self.standardize = standardize
        
        
    def spacy_tokenizer(self, x: str, max_seq_len: int):
        return [w.text for w in SpacyWordSplitter(language='en_core_web_sm',
                                                  pos_tags=False).split_words(x)[:max_seq_len]]
    
    def row_to_instance(self, row):
        tokens = [Token(x) for x in self.tokenizer(row["report"], self.max_seq_len)]
        labels = row[LABEL_COLS].values
        if self.feature_cols:
            features = row[self.feature_cols].values
            return self.text_to_instance(tokens, labels, features)
        
        return self.text_to_instance(tokens, labels)
        
    
    @overrides
    def text_to_instance(self, 
            tokens: List[Token], 
            labels: np.ndarray,
            features: np.ndarray = None
        ) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field} 

        if self.feature_cols:
            feature_field = ArrayField(array=features)
            fields['features'] = feature_field
        
        
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        read_fn = pd.read_json
        if file_path.endswith('.csv'):
            read_fn = pd.read_csv
            print('Using csv reader.')
            
        df = read_fn(file_path)
        df = df.reset_index().rename(columns={'Index': 'id'})
        df = df[df.report.str.split(' ').str.len() > 3]
        df['report'] = clean_text(df['report'], rm_null=False)
        
        if self.feature_cols and self.standardize:
            f_cols = self.feature_cols
            df.loc[:, f_cols] = (df.loc[:, f_cols] - df.loc[:, f_cols].mean()) / df.loc[:, f_cols].std()
            df = df.fillna(0)

        data = df.apply(self.row_to_instance, axis=1)
        return iter(data)
        


def build_vocab(config, reader=None):
    if config.indexer == 'base':
        assert reader

        full_ds = reader.read(DATA_ROOT.format("full_dataset.json"))
        return Vocabulary.from_instances(full_ds, max_vocab_size=max_vocab_size)
    return Vocabulary()


def fit_pipeline(train, featurizer='union', clf='lr'):
    train = train[~train.isnull()]

    train_df = train.drop(columns=['name', 'label'])
    y_train = train.label

    tfidf_pipe = Pipeline([
        ('report_tfidf', Pipeline([
            ('selector', ItemSelector(key='report')),
            ('tfidf', TfidfVectorizer(max_features=10000, strip_accents='unicode')),
        ]))
    ])

    meta = ItemSelector(key=train_df.drop(columns='report').columns)

    features = None
    if featurizer == 'union':
        features = FeatureUnion([
            ('metadata', meta),
            ('tfidf', tfidf_pipe)
        ])
    elif featurizer == 'tfidf':
        features = tfidf_pipe
    elif featurizer == 'metadata':
        features = meta
    else:
        raise ValueError(f'Invalid featurizer: {featurizer}')

    clf_model = CLFS[clf]
    print(
        f'Training {type(clf_model).__name__} with {featurizer} features on {train_df.shape} training set.')

    pipe = Pipeline([
        ('featurizer', features),
        ('impute', SimpleImputer(missing_values=np.nan,
                                 strategy='constant', fill_value=-1)),
        ('scale', StandardScaler(with_mean=False)),
        ('clf', clf_model)
    ]).fit(train_df, y_train)

    return pipe
