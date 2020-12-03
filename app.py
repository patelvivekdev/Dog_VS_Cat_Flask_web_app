from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from matplotlib.pyplot import imread
import tensorflow as tf
import tensorflow_hub as hub

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'

labels = ['Cat', 'Dog']

def load__model():
    """Load model once at running time for all the predictions"""
    print('[INFO] : Model loading ................')
    global model
    model = tf.keras.models.load_model(
        MODEL_FOLDER + '/model.h5', custom_objects={"KerasLayer": hub.KerasLayer})
    print('[INFO] : Model loaded')


def preprocessing_image(path):
    img = imread(path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=[224, 224])
    img = np.expand_dims(img, axis=0)

    return img


def predict(fullpath):
    image = preprocessing_image(fullpath)
    pred = model.predict(image)

    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        # Make prediction
        pred = predict(fullname)
        result = labels[np.argmax(pred)]

        return result
    return None


def create_app():
    load__model()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
