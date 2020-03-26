#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np,os
from flask import Flask, request, jsonify, render_template
import pickle
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import tensorflow as tf
from tensorflow.python.framework import ops

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

global graph
graph = tf.get_default_graph()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload',methods=['POST'])
def upload():
    '''
    For rendering results on HTML GUI
    '''
    for file in request.files.getlist("file"):
        print(file)
        filename=file.filename
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    with graph.as_default():
        prediction = model.predict(img_data)

    return render_template('home.html', prediction_text='Pneumonia is $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)

