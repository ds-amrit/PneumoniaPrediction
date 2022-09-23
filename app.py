# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 03:29:11 2022

@author: amrit
"""

# Importing required libraries
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify,render_template,session
import os
import cv2
from werkzeug.utils import secure_filename




# Defining upload folder path
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')




## CREATING A FLASK WEB APP TO DEPLOY ML MODEL

app = Flask(__name__,template_folder='templateFiles')

## Configuring the path to upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'

#Importing the machine learning model
model = pickle.load(open("D:\delaPlex\image segmentation project\model.pkl",'rb'))

# Defining the decorator for home page
@app.route('/')
def home():
    return render_template('index2.html')
 
# Defining the decorator for predict page
@app.route('/predict',  methods=["POST"])
def predict():
    if request.method == 'POST':
        
        # Upload file flask
        uploaded_img = request.files['uploaded-file']

        # Fetching the file name
        img_filename = secure_filename(uploaded_img.filename)

        # Saving the image to upload folder
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))

        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        
        size=(32,32)
        # Reading the image using opencv
        image_path = cv2.imread(session.get('uploaded_img_file_path', None))

        # Resizing the image as per model requirements
        image = cv2.resize(image_path, size)

        # Creating an image array to feed into model
        image_array = np.array(cv2.resize(cv2.imread(session.get('uploaded_img_file_path', None)), size).flatten()).reshape(1,-1)
        
        # Predicting the result from the model
        result = model.predict(image_array)
        
        # If result =0, that means lungs are healthy 
        if result == 0:
            return render_template('index2.html',prediction_text='It is likely that your lungs are healthy!!')
        else:
            return render_template('index2.html',prediction_text='You might have Pneumonia. Consult a doctor ASAP.')
 


if __name__=='__main__':
    app.run(debug = True)