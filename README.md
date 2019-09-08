# Trouble with the Curve


This repository contains the data, models, and web app for my paper _Trouble with the Curve: Predicting Future MLB Players Using Scouting Reports_. A log of model training runs, including architectures, hyperparameters, and so on, can be found [here](https://ui.neptune.ml/jacobdanovitch/Trouble-with-the-Curve/wiki/README-45a8ee85-4d26-4343-991f-3302fe9ae4d5).

![img](webapp/img/screenshot.png)

## [Data](https://github.com/jacobdanovitch/Trouble-With-The-Curve/tree/master/data)

To the best of my knowledge, this is the only existing dataset of its kind for baseball prospect profiles. The data was acquired from [MLB.com](http://m.mlb.com/prospects/2019)'s top prospect rankings dating back to 2013. This includes rankings for top minor league players, draft prospects, international prospects, and team-specific rankings. Over 5000 player profiles were accumulated, consisting of features such as: 

* Scouting data
  * Written reports, numeric grades, ranking, estimated years away from MLB
* Draft data
  * Overall pick #, team, position, school (if applicable)
* Player IDs for sites like MLB.com, FanGraphs, and Baseball-Reference

## [Models](https://github.com/jacobdanovitch/Trouble-With-The-Curve/tree/master/twtc)

With the above data, an obvious question arises: Can we predict if a player will make the major leagues? We compare the predictive power held in player metadata, written scouting reports, and a combination of the two (reflected in the table below). Our results endorse the role of scouts in Major League Baseball, demonstrating that the written reports are predictive of the future development of young players.

**model**|**accuracy**|**f1**
:-----:|:-----:|:-----:
Logistic Regression|0.8621|0.8311
DNN|0.8477|0.8120
SVM|0.8486|0.8124
SGD|0.8037|0.7555
Ensemble|0.8539|0.8218
BOE|0.7878|0.7346
LSTM|0.8175|0.7736
CNN|**0.8755**|**0.8439**
H-CNN|0.8529|0.8144
_Mean_|_0.8389_|_0.7999_


## [Web App](https://github.com/jacobdanovitch/Trouble-With-The-Curve/tree/master/webapp)

A Hierarchical Attention Network is trained with FastText embeddings for use in an interactive web app, allowing not only a demonstration of the research problem, but also an interpretable visualization for each prediction using attention weights.

As well, the fine-tuned FastText embeddings can be visualized at the [/embeddings](/embeddings) endpoint, using the [Tensorflow Embedding Projector](https://github.com/tensorflow/embedding-projector-standalone/).
