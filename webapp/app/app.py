import json
import requests
import itertools

from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from flask import jsonify
from flask import url_for
from flask import current_app

from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

import torch
import pandas as pd
import numpy as np


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
sentence_splitter = SpacySentenceSplitter()

app = Flask(__name__, static_folder='../')

def i2label(i):
    return 'MLB' if i == 1 else 'MiLB'

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
        
        sentences = sentence_splitter.split_sentences(doc)
        tokenized_sents = (tokenizer.tokenize(sent) for sent in sentences)
        words = [str(w) for w in itertools.chain(*tokenized_sents)]


        res = requests.request("POST", SERVER_URL, data=json.dumps({'sentence': doc}), headers=SERVER_HEADERS).json()
        

        pred = res.get('label')
        if not pred:
            pred = np.argmax(res.get('logits'))
        pred = i2label(int(pred))
        sentence_weights = res.get('sentence_attention')
        word_weights = [w for w in itertools.chain(*res.get('word_attention')) if w != 0]

        print(len(word_weights), len(words))
        
        response = dict(words=words, weights=word_weights, prediction=pred, label=label)
        return jsonify(response)
    return Response(status=501)