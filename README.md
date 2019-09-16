# Trouble with the Curve


This repository contains the data, models, and web app for my paper _Trouble with the Curve: Predicting Future MLB Players Using Scouting Reports_.

![img](webapp/img/screenshot.png)

## [Data](https://github.com/jacobdanovitch/Trouble-With-The-Curve/tree/master/data)

To the best of my knowledge, this is the only existing dataset of its kind for baseball prospect profiles. The data was acquired from [MLB.com](http://m.mlb.com/prospects/2019)'s top prospect rankings dating back to 2013. This includes rankings for top minor league players, draft prospects, international prospects, and team-specific rankings. Over 5000 player profiles were accumulated, consisting of features such as: 

* Scouting data
  * Written reports, numeric grades, ranking, estimated years away from MLB
* Draft data
  * Draft position, team, position, school (if applicable)
* Player IDs for sites like MLB.com, FanGraphs, and Baseball-Reference

## [Models](https://github.com/jacobdanovitch/Trouble-With-The-Curve/tree/master/notebooks)

With the above data, an obvious question arises: Can we predict if a player will make the major leagues? We compare the predictive power held in player metadata, written scouting reports, and a combination of the two (reflected in the table below). Our results endorse the role of scouts in Major League Baseball, demonstrating that the written reports are predictive of the future development of young players.

## [Web App](https://github.com/jacobdanovitch/Trouble-With-The-Curve/tree/master/webapp)

A Hierarchical Attention Network is trained with FastText embeddings for use in an interactive web app, allowing not only a demonstration of the research problem, but also an interpretable visualization for each prediction using attention weights.

As well, the fine-tuned FastText embeddings can be visualized at the [/embeddings](/embeddings) endpoint, using the [Tensorflow Embedding Projector](https://github.com/tensorflow/embedding-projector-standalone/).
