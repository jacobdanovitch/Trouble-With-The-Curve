# Data

## Contents

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

## Included

As of right now, you can find the full dataset in the `twtc.csv` and `twtc.json` files. A sample is provided in `sample.tsv`. Two subfolders, `profile-classification` and `scouting-classification` contain the train/test files used for the experiments outlined in the paper.
