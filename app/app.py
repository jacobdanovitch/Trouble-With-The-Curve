import json

from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from flask import jsonify
from flask import url_for

from analyze import Classifier

import torch
import pandas as pd

word2vec_config_path = "word2vec/fasttext/config.json"
word2vec_model_path = "word2vec/fasttext/word2vec.model"

# Set trained model, config file
HAN_model_id = 2
HAN_mdoel_path = f"word2vec/fasttext/{HAN_model_id}/model1.pwf"
HAN_config_path = f"word2vec/fasttext/{HAN_model_id}/config.json"

# Set tokenizer name
tokenizer_name = "word_tokenizer"

# Set device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Classifier(word2vec_config_path=word2vec_config_path,
                   word2vec_model_path=word2vec_model_path,
                   HAN_mdoel_path=HAN_mdoel_path,
                   HAN_config_path=HAN_config_path,
                   tokenizer_name=tokenizer_name,
                   device=device)

MAX_QUEUE_LEN = 5
queue = []

app = Flask(__name__, static_folder='../')

def clean_text(corp, rm_null=False):
    if rm_null:
        corp = corp[~corp.isnull()]
    corp = corp.str.lower().str.encode('ascii', 'ignore').str.decode('ascii')
    corp = corp.str.replace("bernie pleskoff's scouting report", "")
    corp = corp.apply(lambda doc: ''.join([char if char.isalpha() else f' {char} ' for char in doc]))
    return corp

def i2label(i):
    return 'makes_mlb' if i == 1 else 'minor_leagues'

def flatten(l):
    return [item for sublist in l for item in sublist]

def process_attentions(tokens, word_weights):
    return flatten([[(t, w) for t, w in zip(toks, weights) if w > 0] for toks, weights in zip(tokens, word_weights)])

def cache_queue():
    json_fp = 'app/attentions.json' # url_for('static', filename='app/attentions.json')
    #print('FILEPATH: '+json_fp)
    #import os
    #print(os.listdir())
    #return

    global queue
    with open(json_fp, 'r') as f:
        cache = json.load(f)
    
    for res in queue:
        if res in cache:
            continue
        cache.append(res)
    
    with open(json_fp, 'w') as f:
        f.write(json.dumps(cache, indent=2))
    
    queue = [] 

def update_queue(res):
    global queue
    queue.append(res)
    if len(queue) > MAX_QUEUE_LEN:
        cache_queue()

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/activations')
def activations():
    """
    Receive a text and return HNATT activation map
    """
    if request.method == 'GET':
        label = request.args.get('label')
        doc = request.args.get('text')
        doc = clean_text(pd.Series([doc]))[0]

        top_class, tokens, sent_weights, word_weights = model.analysis(doc)
        sent_weights = sent_weights.tolist()
        word_weights = word_weights.tolist()
        total_len = len(sent_weights)

        words, weights = zip(*process_attentions(tokens, word_weights))
        pred = i2label(top_class.item())
        
        response = dict(words=words, weights=weights, prediction=pred, label=label)
        # update_queue(response)
        return jsonify(response)
    return Response(status=501)