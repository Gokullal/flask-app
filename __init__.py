#!/usr/bin/env python3
import cgi
import cgitb
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from pickle import load
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import keras
import keras.utils as image
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
import tensorflow as tf
import string
import requests
from PIL import Image
from io import BytesIO
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Enable detailed error messages
cgitb.enable()

# Load the trained model to classify sign
base_model = InceptionV3(
    weights='inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
vgg_model = Model(base_model.input, base_model.layers[-2].output)


def preprocess_img_from_url(url):
    # Inception V3 expects images in 299x299 size
    response = requests.get(url, verify=False)
    img = load_img(BytesIO(response.content), target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess_img_from_url(image)
    vec = vgg_model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec


def greedy_search(pic, wordtoix, max_length):
    # Your implementation for greedy search goes here
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# Load the model (new-model-1.h5) for generating captions
model = load_model('new-model-1.h5')

# Load wordtoix and ixtoword from pickle files
with open('wordtoix.pkl', 'rb') as f:
    wordtoix = load(f)

with open('ixtoword.pkl', 'rb') as f:
    ixtoword = load(f)


############## Semantic model ################


def feature_extractions(directory, image_url):
    model = tf.keras.applications.vgg16.VGG16()
    model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    url= image_url
    image_response = requests.get(url)
    img = Image.open(BytesIO(image_response.content))
    img = img.resize((224, 224))  # Resize the image to match the VGG16 input size
    
    arr = keras.utils.img_to_array(img, dtype=np.float32)
    arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
    arr = keras.applications.vgg16.preprocess_input(arr)
    
    features = model.predict(arr, verbose=0)
    return features
def sample_caption(model, tokenizer, max_length, vocab_size, feature):
    caption = "<startseq>"
    while 1:
        encoded = tokenizer.texts_to_sequences([caption])[0]
        padded = pad_sequences([encoded], maxlen=max_length, padding='pre')[0]
        padded = padded.reshape((1, max_length))
        pred_Y = model.predict([feature, padded])[0,-1,:]
        next_word = tokenizer.index_word[pred_Y.argmax()]
        caption = caption + ' ' + next_word
        if next_word == '<endseq>' or len(caption.split()) >= max_length:
            break
    caption = caption.replace('<startseq> ', '')
    caption = caption.replace(' <endseq>', '')
    return(caption)

def funSemantic(image_url):
    with open('./semantic/File1.json', 'r') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    model = keras.models.load_model("./semantic/File2.h5")
    vocab_size = tokenizer.num_words
    max_length = 37
    img = image_url
    features = feature_extractions("./semantic", img)
    filename = "img1.jpg"  # This should be the filename or identifier of the image
    # plt.figure()
    caption = sample_caption(model, tokenizer, max_length, vocab_size, features)
    img = tf.keras.utils.load_img("./semantic/" + filename)
    return(caption)


############## End Semantic model ################

# Create a Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True)


# Define your Flask route to process the image


@app.route('/', methods=['POST'])
def process_image_route():
    data = request.get_json()
    image_url = data["image_url"]
    enc = encode(image_url)
    image = enc.reshape(1, 2048)

    # Define max_length (adjust this value based on your sequence length)
    max_length = 74

    # Example usage for generating captions using greedy search
    pred = greedy_search(image, wordtoix, max_length)

    semantic_caption = funSemantic(image_url)








    # Create a dictionary to store the caption
    result = {"caption": pred,"semantic": semantic_caption}
    # Return the JSON response
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
