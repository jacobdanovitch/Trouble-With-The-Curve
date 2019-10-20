import os, subprocess

os.environ['FLASK_APP'] = "app/app.py"
os.environ['FLASK_DEBUG'] = "1"

# for LSTM instead: https://storage.googleapis.com/jacobdanovitch/twtc/lstm.tar.gz
# Will have to change app.py to accept only attention_weights


subprocess.call('./serve_model.sh')
subprocess.call(['flask', 'run'])
