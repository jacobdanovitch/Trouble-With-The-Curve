import json
import requests

from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from flask import jsonify
from flask import url_for
from flask import current_app

from allennlp.data.tokenizers.word_tokenizer import WordTokenizer

import torch
import pandas as pd


SERVER_URL = "http://localhost:8000/predict"
SERVER_HEADERS = {
    'Content-Type': "application/json",
    'User-Agent': "PostmanRuntime/7.17.1", #TODO: Set user agent
    'Accept': "*/*",
    'Cache-Control': "no-cache",
    'Host': "localhost:8000",
    'Accept-Encoding': "gzip, deflate",
    'Content-Length': "40",
    'Connection': "keep-alive",
    'cache-control': "no-cache"
}

tokenizer = WordTokenizer()

app = Flask(__name__, static_folder='../')

def i2label(i):
    return 'makes_mlb' if i == 1 else 'minor_leagues'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return "Coming soon"
    

@app.route('/embeddings')
def embeddings():
    return current_app.send_static_file('projector-master/embeddings.html')

@app.route('/activations')
def activations():
    """
    Receive a text and return HNATT activation map
    """
    if request.method == 'GET':
        label = request.args.get('label')
        doc = request.args.get('text')
        words = [str(w) for w in tokenizer.tokenize(doc)]

        res = requests.request("POST", SERVER_URL, data=json.dumps({'sentence': doc}), headers=SERVER_HEADERS).json()

        pred = i2label(int(res['label']))
        weights = res['attention_weights']
        
        response = dict(words=words, weights=weights, prediction=pred, label=label)
        return jsonify(response)
    return Response(status=501)