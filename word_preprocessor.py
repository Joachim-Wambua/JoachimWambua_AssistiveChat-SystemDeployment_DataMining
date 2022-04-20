# Code for Preprocessing user text input

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import random
import pickle
import numpy as np
from keras.models import load_model


# Import Assistive Chatbot Model
chatbot_model = load_model('models/optimised_chatbot_model.h5')


# Initialise Necessary modules
intents = json.loads(open('disabilities_intent.json', encoding='utf-8').read())
words = pickle.load(open('models/words.pkl','rb'))
output_classes = pickle.load(open('models/output_tags.pkl','rb'))


# Function to Preprocess user input
def preprocess_input_text(user_input):
    input_words = nltk.word_tokenize(user_input)
    input_words = [lemmatizer.lemmatize(w.lower()) for w in input_words]
    return input_words


# Function to evaluate bag of words (a representation of text that describes the occurrence of words within a document)
def bag_of_words(user_input, words, view_details = True):
    # Tokenizing & Lemmatizing User Input
    input_words = preprocess_input_text(user_input)

    # Bag of Words - Vocabulary Matrix
    bag = [0]*len(words)

    for i in input_words:
        for j, k in enumerate(words):
            if i == k:
                # Assign 1 if current word is in intent vocabulary
                bag[j] = 1
                if view_details:
                    print(f"Found in Bag: {k}")
    return(np.array(bag))


# Function to predict user intent
def predictor(model, user_input):
    # Filter predictions
    err_threshold = 0.25
    to_predict = bag_of_words(user_input, words, view_details=False)
    prediction = model.predict(np.array([to_predict]))[0]

    results = [[m,n] for m,n in enumerate(prediction) if n > err_threshold]

    # Sort by probability values
    results.sort(key=lambda y: y[1], reverse=True)
    result_arr = list()

    # Loop through results & display their probability values
    for result in results:
        result_arr.append({"user_intent": output_classes[result[0]], "probability": str(result[1])})

    return result_arr


# Function to generate responses to the user
def generate_response(intent_file, ints):
    tag = ints[0]['user_intent']
    intent_list = intent_file['intents']

    for i in intent_list:
        if(i['tag'] == tag):
            response = random.choice(i['responses'])
            break
        else:
            response = "I don't understand your questions! Kindly ask a valid question"
    return response


# Function to print response
def chatbot_response(user_input):
    prediction = predictor(chatbot_model, user_input)
    response = generate_response(intents, prediction)
    return response