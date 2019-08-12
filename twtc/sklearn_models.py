from data_utils import load_df
from constants import CLFS
from config import Config

import argparse

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

import neptune as nt


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
    

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--clf', type=str)
    parser.add_argument('-f', '--features', type=str)

    parser.add_argument('-t', '--track', action='store_true')
    
    return parser.parse_args()

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
  print(f'Training {type(clf_model).__name__} with {featurizer} features on {train_df.shape} training set.')

  pipe = Pipeline([
      ('featurizer', features),
      ('impute', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)),
      ('scale', StandardScaler(with_mean=False)),
      ('clf', clf_model)
  ]).fit(train_df, y_train)
  
  return pipe


if __name__ == '__main__':
    args = build_argparser()
    config = Config(tags=['sklearn', args.features, args.clf], **vars(args))
    config.parameter('max_tfidf_features', 10_000)
    
    train = load_df('train')
    pipe = fit_pipeline(train, args.features, args.clf)

    test = load_df('test')
    preds = pipe.predict(test.drop(columns=['name', 'label']))
    
    truth = test.label.values

    acc = balanced_accuracy_score(truth, preds)
    f1 = f1_score(truth, preds, average='binary')

    print(f'Balanced accuracy\n---\n{acc:.4f}\n')
    print(f'Binary F-1\n---\n{f1:.4f}\n')

    print('Classification report\n---')
    print(classification_report(truth, preds, target_names=['minors', 'mlb']))

    if config.track:
        config.log('acc', acc)
        config.log('f1', f1)

        nt.stop()

