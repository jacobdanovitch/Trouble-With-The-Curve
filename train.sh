export LOC="twtc" #"twtc"
python -m allennlp.run train \
                        $LOC/model_configs/biattentive_classification_network.jsonnet \
                        --include-package $LOC \
                        -s $LOC/savedmodels