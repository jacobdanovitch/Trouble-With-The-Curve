cd jdnlp;

python -m allennlp.service.server_simple \
                --archive-path  "https://storage.googleapis.com/jacobdanovitch/twtc/han_lstm.tar.gz" \
                --predictor text_classifier \
                --include-package jdnlp \
                --field-name sentence \
                --title "Trouble with the Curve" &