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

# Create a Flask app
app = Flask(__name__)
CORS(app)


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

    # Create a dictionary to store the caption
    result = {"caption": pred}

    # Return the JSON response
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,
            ssl_context=('server.crt', 'server.key'))
