# Trouble with the Curve

This folder contains the code required for training the models described in the paper. All deep learning models were assembled using `AllenNLP`, and can be run with `python train.py`. Training logs can be viewed [here](https://ui.neptune.ml/jacobdanovitch/Trouble-with-the-Curve/experiments).

## Report classification

Using only the contents of the written scouting report, the models performed as follows:

**model**|**f-1**|**accuracy**
:-----:|:-----:|:-----:
CNN|**0.8297**|0.5354
Highway-CNN|0.7987|0.5224
LSTM|0.7680|0.5386
Bag-of-Embeddings|0.5986|**0.5696**

## Profile classification

A consistent (though marginal) improvement was seen when integrating metadata such as how high the player was drafted, their MLB-readiness, their ranking, and a variety of numeric grades for traits such as hitting, running, throwing, and so on.

_<tables to be added shortly>_

## Upcoming additions

* CLI documentation
* Sk-learn models
* Training with additional metadata as well as text features
