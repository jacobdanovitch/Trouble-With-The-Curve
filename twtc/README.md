# Trouble with the Curve

This folder contains the code required for training the models described in the paper. All deep learning models were assembled using `AllenNLP` (using [this tutorial](https://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/) as reference) and `scikit-learn`. Training logs can be viewed [here](https://ui.neptune.ml/jacobdanovitch/Trouble-with-the-Curve/experiments).

We performed experiments with three different feature sets to predict if a player would make the MLB: metadata, a written scouting report, and a union of the two.

## Metadata classification

Here, we used metadata for each prospect to classify if they would make the MLB. This metadata consisted of features like:

* Ranking on their prospect list
* The spot at which they were drafted
* How major-league ready they were (how many years scouts estimated it would take for them to reach the majors)
* Numeric grades for a variety of skills
  * Contact
  * Power
  * Speed
  * Fielding
  * Throwing
  * Pitching
    * Control
    * Fastball
    * Changeup
    * Curveball
    * Slider
    * Cutter
    * Splitter
  * Overall

Several models were evaluated using `scikit-learn`. The results are as follows:

**model**|**accuracy**|**f1**
:-----:|:-----:|:-----:
Logistic Regression|0.6621|0.5572
DNN|0.6620|0.5631
SVM|0.6735|0.5797
SGD|**0.6777**|**0.6058**
Ensemble|0.6298|0.4807
_Mean_|_0.6610_|_0.5573_

Stochastic gradient descent performs best when classifying the tabular data, though the results are overall fairly poor. This suggests that typical measures such as rankings, draft position, and scouting grades are insufficient to reliably predict if a prospect will make the major leagues.

## Report classification

We then compared the above results to performance when using only the contents of the written scouting report. We added several deep-learning based models for natural language processing, using `AllenNLP`. For the `scikit-learn` models, we featurized each report using `Tf-Idf` with a maximum vocabulary size of 10,000.

<i style='font-size: small'>Note: 'BOE' is short for 'Bag of Embeddings'</i>

**model**|**accuracy**|**f1**
:-----:|:-----:|:-----:
Logistic Regression|0.8533|0.8193
DNN|0.8393|0.8020
SVM|0.8490|0.8122
SGD|0.8327|0.7924
Ensemble|0.8466|0.8112
BOE|0.7908|0.7340
LSTM|0.7935|0.7379
CNN|**0.8689**|**0.8318**
H-CNN|0.8525|0.8075
_Mean_|_0.8363_|_0.7942_

The Convolutional Neural Network (CNN) performs best at predicting if a player will make the major leagues, based on only their scouting report. While traditionally an LSTM would likely be most performant on text classification, we hypothesize that the length of the reports (generally a paragraph at a minimum) make it challenging for the LSTM to learn as effectively. In contrast, CNNs could benefit from the style in which the reports are written (generally segmented into sections, the order of which is not necessarily important). 

## Profile classification

A consistent, though very slight, improvement was seen when combining both sets of features.

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

As is to be expected, the majority of models see improved performance when using the full set of profile data. That said, the improvement was marginal at best, suggesting that the written reports are far more predictive than the aforementioned metadata. 

These findings provide a light endorsement of the role of professional scouts in Major League Baseball, suggesting that their written observations hold predictive power in the future development of young players.

## Future Work

While the performance of numeric features were studied with [metadata](#Metadata-classification), further research should be done to integrate additional statistical features. In our full dataset, we provide player IDs to join each record to the corresponding player on popular resources such as MLB.com, FanGraphs, and Baseball-Reference. Future works should explore the integration of player statistics from these resources using the IDs, and examine their predictive power as well. Likewise, in lieu of predicting if a player will simply make the MLB, more finely-grained objectives could be considered (such as predicting future Wins Above Replace (WAR), by regression or binning and classification).


