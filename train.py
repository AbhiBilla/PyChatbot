#Importing Required Packages
import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

#Creating json file
a={"intents": [
    {"tag": "Basic1",
     "patterns": ["Python literals", "literals in python", "literals python?"],
     "responses": ['''String Literals: It is a sequence of characters enclosed in codes.
                             There can be single, double and triple strings based on the number of quotes used.
                             Character literals are single characters surrounded by single or double-quotes. 
                             Numeric Literals: These are unchangeable kind and belong to three different types – integer, float and complex.
                             Boolean Literals: They can have either of the two values- True or False which represents ‘1’ and ‘0’ respectively. 
                             Special Literals: Special literals are sued to classify fields that are not created. It is represented by the value ‘none’. ''']
    },
        {"tag": "greeting",
     "patterns": ["Hi", "Hey", "Is anyone there?", "Hello", "Hay"],
     "responses": ["Hello", "Hi", "Hi there"]
    },
    {"tag": "goodbye",
     "patterns": ["Bye", "See you later", "Goodbye"],
     "responses": ["See you later", "Have a nice day", "Bye! Come back again"]
    },
    {"tag": "work",
     "patterns": ["What do you do", "How will you assist me?", "What work will you do?"],
     "responses": ["Iam an AI bot. I'll help you in answering your python questions"]
    },
    {"tag": "thanks",
     "patterns": ["Thanks", "Thank you", "That's helpful", "Thanks for the help"],
     "responses": ["Happy to help!", "Any time!", "My pleasure", "You're most welcome!"]
    },
    {"tag": "about",
     "patterns": ["Who are you?", "What are you?", "Who you are?" ],
     "responses": ["I.m Joana, your bot assistant", "I'm Joana, an Artificial Intelligent bot"]
    },
    {"tag": "name",
    "patterns": ["what is your name", "what should I call you", "whats your name?"],
    "responses": ["You can call me Joana.", "I'm Joana!", "Just call me as Joana"]
    },
    {"tag": "Basic2",
     "patterns": ["Python keywords", "Keywords in python", "keywords in python language"],
     "responses": ['''Keywords in Python are reserved words which are used as identifiers, function name or variable name.
                            They help define the structure and syntax of the language.
                            Eg : 1. except
                                    2. break''']
    },
    {"tag" : "Basic3",
     "patterns" : ["paradigms in python","python paradigms","paradigms python support"],
     "responses" : ['''Python supports three types of Programming paradigms
                            Object Oriented programming paradigms 
                            Procedure Oriented programming paradigms
                            Functional programming paradigms ''']
        
    },
    {"tag": "Medium1",
     "patterns": ["Lambda python", "Lambda Function", "Lambda in python", "python lambda function"],
     "responses": [''' An anonymous function is known as a lambda function.
                           This function can have any number of parameters but, can have just one statement.
                            Eg : a = lambda x,y : x+y
                                   print(a(5, 6))
                            Output: 11''']
    },
    {"tag": "Medium2",
     "patterns": ["Pickling in python", "Unpickling in python", "Pickiling and unpickling" ],
     "responses": ['''Pickle module accepts any Python object and converts it into a string representation and dumps it into a file by using dump function, this process is called pickling.
                            While the process of retrieving original Python objects from the stored string representation is called unpickling. ''']
    },
    {"tag": "Medium3",
    "patterns": ["Python libraries", "libraries in python", "Name few libraries in python"],
    "responses": ['''Python libraries are a collection of Python packages.
                           Some of the majorly used python libraries are – Numpy, Pandas, Matplotlib, Scikit-learn and many more. ''']
    },
    {"tag": "Hard1",
    "patterns": ["lists or numpy array", "Is numpy array faster than lists", "differences of lists and numpy arrays", "lists and numpy arrays", "numpy arrays or lists"],
    "responses": [''' We use python numpy array instead of a list because of the below three reasons:

Less Memory
Fast
Convenient ''']
    },
    {"tag": "Hard2",
    "patterns": ["Dataframes combine", "how to combine dataframes in python", "combining dataframes in python", "function for combining dataframes"],
    "responses": ["Two different data frames can be stacked either horizontally or vertically by the concat(), append() and join() functions in pandas."]
    },
    {"tag": "Hard3",
    "patterns": ["init in python", "init function in python", "use of init function"],
    "responses": ['''_init_ methodology is a reserved method in Python aka constructor in OOP. 
                           When an object is created from a class and _init_ methodolgy is called to acess the class attributes.''']
    }
]
}
s = json.dumps(a)
with open("intents.json","w") as f:
  f.write(s)

# Loading File
with open('intents.json') as file:
    data = json.load(file)
    
training_sentences = []
training_labels = []
labels = []
responses = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

#Model Training
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()

#Model Fit
epochs = 1000
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

#Saving Model
model.save("Chatbot_py.hdf5")

import pickle

# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)