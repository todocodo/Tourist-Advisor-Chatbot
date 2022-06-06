from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import random
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# import tensorflow as tf

# Create a Flask Instance
app = Flask(__name__)

history = []

# All pre-defined responeses 
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "What's up", "Hola hola"]

GOODBYE_RESPONSES = ["bye", "Goodbye", "See ya", "It was lovely chatting with you! Bye.", "Bye, till next time!"]

TRAVEL_SUGGESTION_RESPONSES = ["Where would you like to go - a lovely restaurant or a nice bar maybe", "I know a lot of restaurants and bars, tell me what you want to do?"]

APOLOGISE_RESPONSES = ["Sorry, can you repeat, please", "Say it again", "Bro, I can't understand you", "Sorry, I didn't get that"]

YOU_ARE_WELCOME_RESPONSES = ["You are very welcome", "Happy to help", "No problem", "No worries", "That's my job", "You're welcome"]

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

# Create a route decorator which also handles the main method for the chatbot

@app.route('/', methods=["GET","POST"]) 

def index(): 
  textInput = request.form.get("textInput")

  if textInput:

    bot = TouristBotClassifier(classes,model,tokenizer,label_encoder)  
    pred_results = bot.get_intent(textInput)

    global history 
    history = dialogue_management(history, pred_results, textInput)
  
  if textInput == "clear":
    history = []

  return render_template("index.html", history=history)


def dialogue_management(history, pred_results, textInput):
  if pred_results == "greeting":
    history.append([textInput, random.choice(GREETING_RESPONSES)])
  elif pred_results =="goodbye":
    history.append([textInput, random.choice(GOODBYE_RESPONSES)])
  elif pred_results == "travel_suggestion":
    history.append([textInput, random.choice(TRAVEL_SUGGESTION_RESPONSES)])
  elif pred_results == "restaurant_suggestion":
    history.append([textInput, "Go Five Guys, it is the best place to eat some burgers!"])
  elif pred_results == "bar_suggestion":
    history.append([textInput, "Try Pop World, the music is great and also the cocktails"])
  elif pred_results == "thank_you":
    history.append([textInput, random.choice(YOU_ARE_WELCOME_RESPONSES)])
  elif pred_results == "oss":
    history.append([textInput, "I am not answering to that!"])
  elif pred_results == "weather":
    history.append([textInput, "I only know the weather in Burundi. It's around 36 degrees"])
  else:
    history.append([textInput, random.choice(APOLOGISE_RESPONSES)])
  return history

