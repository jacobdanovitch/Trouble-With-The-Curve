import multiprocessing as mp
from tqdm.autonotebook import tqdm; tqdm.pandas()

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

def onehot_encode_column(df, col):
    enc = OneHotEncoder()
    encodings = enc.fit_transform(df[col].values.reshape(-1, 1)).todense()
    
    names = [f.replace('x0_', '') for f in enc.get_feature_names()]
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
    try:
        chunked = nltk.ne_chunk(nltk.tag.pos_tag(nltk.word_tokenize(txt)))
    except:
        raise ValueError(txt)
    subs = {" ".join(w for w, t in elt): elt.label() for elt in chunked if isinstance(elt, nltk.Tree)}
    return replace(txt, subs)


# https://datascience.blog.wzb.eu/2017/06/19/speeding-up-nltk-with-parallel-processing/
def apply_text_mask(reports, processes=2):
    masked = []
    with mp.Pool(processes=processes) as pool:
            with tqdm(total=len(reports)) as pbar:
                for i, x in enumerate(pool.imap_unordered(mask_text, reports)):
                    pbar.update()
                    masked.append(x)
    return masked




class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]