# Trouble with the Curve

![img](../notebooks/assets/img/ui.jpg)

This web app accompanies my upcoming paper, _Trouble with the Curve: Predicting Future Major-Leaguers from Scouting Reports_.

The data was acquired from [MLB.com](http://m.mlb.com/prospects/2019) and [FanGraphs](fangraphs.com). A [Hierchical Attention Network](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf) implemented with [allennlp](https://github.com/allenai/allennlp/) was trained to classify each report. You can find the implementation in my personal text processing library [here](https://github.com/jacobdanovitch/jdnlp). The attention weights for each classification were then visualized using [yuhaozhang's](https://github.com/yuhaozhang/text-attn-vis) excellent visualization tool.

To run the webapp, simply execute `run.py`. Note that you'll have to clone my library as a submodule first.