from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import json

#CORS
from flask_cors import CORS

# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './bird_species_mobilenet.h5'

# Load your trained model
model = load_model(MODEL_PATH)
  

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

with open('./image_classes.json','r') as classes_input:
  image_class = json.load(classes_input,object_hook=jsonKeys2int)       


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    return answer


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        probability=np.argmax(preds,axis=1)
        mypred=image_class[probability[0]]
        return mypred
    return None


if __name__ == '__main__':
    CORS(app)
    app.run(debug=False)

