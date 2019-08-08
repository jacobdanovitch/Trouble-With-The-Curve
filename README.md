# Trouble with the Curve


This repository contains the data, models, and web app for my paper _Trouble with the Curve: Predicting Future MLB Players Using Scouting Reports_. A log of model training runs, including architectures, hyperparameters, and so on, can be found [here](https://ui.neptune.ml/jacobdanovitch/Trouble-with-the-Curve/wiki/README-45a8ee85-4d26-4343-991f-3302fe9ae4d5).

![img](webapp/img/screenshot.png)

## [Data](https://github.com/jacobdanovitch/Trouble-With-The-Curve/tree/master/data)

To the best of my knowledge, this is the only existing dataset of its kind for baseball prospect profiles. The data was acquired from [MLB.com](http://m.mlb.com/prospects/2019)'s top prospect rankings dating back to 2013. This includes rankings for top minor league players, draft prospects, international prospects, and team-specific rankings. Over 5000 player profiles were accumulated, consisting of features such as: 

* Scouting data
  * Written reports
  * Numeric grades
  * Ranking
  * ETA
* Draft data
  * Overall pick #
  * Team
  * Position
  * School
* Player IDs for sites like MLB.com, FanGraphs, and Baseball-Reference

## [Models](https://github.com/jacobdanovitch/Trouble-With-The-Curve/tree/master/twtc)

With the above data, an obvious question arises: Can we predict if a player will make the major leagues using their scouting report? Using only the written reports:

**model**|**f-1**|**accuracy**
:-----:|:-----:|:-----:
CNN|**0.8297**|0.5354
Highway-CNN|0.7987|0.5224
LSTM|0.7680|0.5386
Bag-of-Embeddings|0.5986|**0.5696**


## [Web App](https://github.com/jacobdanovitch/Trouble-With-The-Curve/tree/master/webapp)

A Hierarchical Attention Network is trained with FastText embeddings for use in an interactive web app, allowing not only a demonstration of the research problem, but also an interpretable visualization for each prediction using attention weights.

As well, the fine-tuned FastText embeddings can be visualized at the [/embeddings](/embeddings) endpoint, using the [Tensorflow Embedding Projector](https://github.com/tensorflow/embedding-projector-standalone/).
