from typing import *

from .constants import DATA_ROOT, USE_GPU

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

from allennlp.models import Model
from allennlp.data.iterators import DataIterator, BasicIterator
from allennlp.nn import util as nn_util
from allennlp.data import Instance

from tqdm.autonotebook import tqdm
from scipy.special import expit # the sigmoid function

import torch

import pandas as pd
import numpy as np

def tonp(tsr): 
  return tsr.detach().cpu().numpy()
 
class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
         
    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return expit(tonp(out_dict["class_logits"]))
     
    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)


def calculate_metrics(model, test_ds, vocab, batch_size=128):
    # iterate over the dataset without changing its order
    seq_iterator = BasicIterator(batch_size=batch_size)
    seq_iterator.index_with(vocab)
    
    predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1)
    test_preds = predictor.predict(test_ds) 

    test_df = pd.read_json(DATA_ROOT.format('test.json'))

    preds = (test_preds > .5).astype('int') 
    truth = test_df.label.values

    cm = confusion_matrix(truth, preds) / len(truth)

    print(f'Accuracy: {(preds == truth).mean()}')
    print(f'F-1: {f1_score(preds, truth, average="binary")}')

    sns.set(font_scale=1.25)
    labels = ['minors', 'mlb']

    sns.heatmap(cm, annot=True, annot_kws={"size": 16}, xticklabels=labels, yticklabels=labels)

    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')

    return plt