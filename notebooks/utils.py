import pandas as pd

import multiprocessing as mp
from tqdm.auto import tqdm; tqdm.pandas()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

import re
import nltk

#nltk.download('maxent_ne_chunker')
#nltk.download('words')

dropped_cols = [
    'expected',
    'birthdate',
    'year',
    'eta',
    'mlb_played_first',
    'debut_age',
    'source',
    'Name',
    'Season',
    'Age',
    'Team',
    'Name_h',
    'Season_h',
    'Team_h',
    'Age_h',
    'Level_h',
    'uid',
    'PlayerId',
    'PlayerId_h'
]

def estimate_eta(age):
    if pd.isnull(age):
        return age
    return (0.5 * age) + 12.757
    


def onehot_encode_column(df, col):
    enc = OneHotEncoder()
    encodings = enc.fit_transform(df[col].values.reshape(-1, 1)).todense()
    
    names = [f.replace('x0', 'pos') for f in enc.get_feature_names()]
    encodings = pd.DataFrame(encodings, columns=names)
    
    assert len(df) == len(encodings)
    
    df = pd.concat([df.drop(columns=col), encodings.astype(int)], axis=1)
    return df


# https://gist.github.com/carlsmith/b2e6ba538ca6f58689b4c18f46fef11c
def replace(string, substitutions):
    if not substitutions:
        return string
    
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    
    return regex.sub(lambda match: substitutions[match.group(0)], string)

def mask_text(txt):
    if not txt or pd.isnull(txt):
        return txt
    chunked = nltk.ne_chunk(nltk.tag.pos_tag(nltk.word_tokenize(txt)))
    subs = {" ".join(w for w, t in elt): elt.label() 
            for elt in chunked 
            if isinstance(elt, nltk.Tree)}
    return replace(txt, subs)


# https://datascience.blog.wzb.eu/2017/06/19/speeding-up-nltk-with-parallel-processing/
# https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
def tqdm_parallel(fn, vals, processes):
    with mp.Pool(processes=processes) as pool, tqdm(total=len(vals)) as pbar:
        for x in pool.imap(fn, vals):
            pbar.update()
            yield x

def apply_text_mask(reports, processes=2):
    return list(tqdm_parallel(mask_text, reports, processes))


def str2float(s):
    if s.replace('.','',1).isdigit():
        return float(s)
    return s

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
    
    
    