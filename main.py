from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# import tensorflow as tf

# Create a Flask Instance
app = Flask(__name__)

history = []

# Load the model & utilies - make sure the libraries installed are the same version
# as the ones used for training the model 

model = load_model('model/model_final.h5')

classes = pickle.load(open('model/classes_final.pkl','rb'))
tokenizer = pickle.load(open('model/tokenizer_final.pkl','rb'))
label_encoder = pickle.load(open('model/label_encoder_final.pkl','rb'))

# Create a route decorator 
@app.route('/') 

def index(): 

  return render_template("index.html")

# Create a predict page
@app.route('/chatbot') 

def predict(): 
    return render_template("chatbot.html")

# In this method the text from the form needs to be preprocessed 
# It should be done the same way as in the notebooks that we made 

@app.route('/chatbot', methods=["POST"]) 

def results(): 
    textInput = request.form.get("textInput")

    kerasas = tokenizer.texts_to_sequences(textInput)
    keras_sequence = pad_sequences(kerasas, maxlen=16, padding="post")
    predictions = model.predict(keras_sequence)
 
    pred_results = label_encoder.inverse_transform(np.argmax(predictions, 1))[0]

    history.append("User: {} ---> BOT_Intent: {}".format(textInput, pred_results))

    return render_template("chatbot.html", history=history)