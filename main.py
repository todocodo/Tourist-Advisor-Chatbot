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

# Create a class for the bot
class TouristBotClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return label_encoder.inverse_transform(np.argmax(self.pred,1))[0]

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

# def results(): 
#     textInput = request.form.get("textInput")

#     kerasas = tokenizer.texts_to_sequences(textInput)
#     keras_sequence = pad_sequences(kerasas, maxlen=16, padding="post")
#     predictions = model.predict(keras_sequence)
 
#     pred_results = label_encoder.inverse_transform(np.argmax(predictions, 1))[0]

#     history.append("User: {} ---> BOT_Intent: {}".format(textInput, pred_results))

#     return render_template("chatbot.html", history=history)


def results(): 

  bot = TouristBotClassifier(classes,model,tokenizer,label_encoder)

  textInput = request.form.get("textInput")

  pred_results = bot.get_intent(textInput)

  history.append("User: {} ---> BOT_Intent: {}".format(textInput, pred_results))

  return render_template("chatbot.html", history=history)