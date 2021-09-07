#Installing Packages
pip install colorama

#Code for chatbot
import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model('Chatbot_py.hdf5')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    score = 0
    count = 0

    
    while True:
        if count==3:
          count = 0
          avg = score/3.0
          print(" ")
          print(" ")
          if (avg>0) and (avg<1.5):
            print("----->>>>>User Level - 1<<<<<------")
          if (avg>1.5) and (avg<2.3):
            print("----->>>>>User Level - 2<<<<<------")
          if (avg>2.3) and (avg<=3):
            print("----->>>>>User Level - 3<<<<<------")
            

        print(Fore.LIGHTBLUE_EX + "User:  " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
          if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot: " + Style.RESET_ALL , np.random.choice(i['responses']))
                if i['tag']=='Basic2':
                  count = count + 1
                  score = score + 1
                  print(count)
                if i['tag']=='Basic1':
                  count = count + 1
                  score = score + 1
                  print(count)
                if i['tag']=='Basic3':
                  count = count + 1
                  score = score + 1
                  print(count)
                if i['tag']=='Medium1':
                  count = count + 1
                  score = score + 2
                  print(count)
                if i['tag']=='Medium2':
                  count = count + 1
                  score = score + 2
                  print(count)
                if i['tag']=='Medium3':
                  count = count + 1
                  score = score + 2
                  print(count)
                if i['tag']=='Hard1':
                  count = count + 1
                  score = score + 3
                  print(count)
                if i['tag']=='Hard2':
                  count = count + 1
                  score = score + 3
                  print(count)
                if i['tag']=='Hard3':
                  count = count + 1
                  score = score + 3
                  print(count)
        if count==3:
          count = 0
          avg = score/3.0
          score = 0
          print(" ")
          print(" ")
          if (avg>0) and (avg<1.5):
            print("----->>>>>User Level - 1<<<<<------")
          if (avg>1.5) and (avg<2.3):
            print("----->>>>>User Level - 2<<<<<------")
          if (avg>2.3) and (avg<=3):
            print("----->>>>>User Level - 3<<<<<------")
                
                  



        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()