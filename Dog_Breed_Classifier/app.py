import sys
import os
clear = lambda: os.system('cls')
clear()
import flask
from flask import render_template, redirect
from flask import request, url_for, render_template, redirect,session
import io
import os
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from keras.utils import np_utils
from glob import glob
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# extract pre-trained face detector
tf.keras.backend.clear_session()
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
from tensorflow.keras.applications.resnet50 import ResNet50
# define ResNet50 model



from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    from keras.preprocessing import image
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    tf.keras.backend.clear_session()
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']
RESNET50_model = Sequential()
RESNET50_model.add(GlobalAveragePooling2D(input_shape=( 7, 7, 2048)))
RESNET50_model.add(Dense(133, activation='softmax'))
RESNET50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
'''Room for training if needed'''
RESNET50_model.load_weights('saved_models/weights.best.RESNET50.hdf5')
dog_names = eval(open("dict.txt").read())
from extract_bottleneck_features import *

def RESNET50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = RESNET50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def final_alg(path):
    tf.keras.backend.clear_session()
    #import matplotlib.pyplot as plt
    #import matplotlib.image as mpimg
    #img = mpimg.imread(path)
    #plt.imshow(img)
    #plt.show()
    if face_detector(path) ==True:
        tf.keras.backend.clear_session()
        predicted_breed = RESNET50_predict_breed(path)
        predicted_breed = predicted_breed[15:]
        print("You are human\nYour resembling dog breed is:",predicted_breed)
        message = "You are human\nYour resembling dog breed is:"+str(predicted_breed)
        return message
    elif dog_detector(path) ==True:
        tf.keras.backend.clear_session()
        predicted_breed = RESNET50_predict_breed(path)
        predicted_breed = predicted_breed[15:]
        print("You are a dog\nYour breed is:",predicted_breed)
        message = "You are a dog\nYour breed is:"+str(predicted_breed)
        return message
    else:
        print("You are neither human nor dog")
        message = "You are neither human nor dog"
        return message








#ignore AVX AVX2 warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

UPLOAD_FOLDER = os.path.join(app.root_path,'static','img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["POST", "GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": 'Upload picture to classify and get dog breed'}
    title = "Upload picture to classify and get dog breed"
    name = "default.png"
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):

            image1 = flask.request.files["image"]
            # save the image to the upload folder, for display on the webpage.
            image = image1.save(os.path.join(app.config['UPLOAD_FOLDER'], image1.filename))
            path = os.path.join(app.config['UPLOAD_FOLDER'], image1.filename)

            # read the image in PIL format
            with open(os.path.join(app.config['UPLOAD_FOLDER'], image1.filename), 'rb') as f:
                image = Image.open(io.BytesIO(f.read()))



            # classify the input image and then initialize the list
            # of predictions to return to the client

            results = final_alg(path)
            data["predictions"] = results

            # loop over the results and add them to the list of
            # returned predictions


            # indicate that the request was a success
            data["success"] = "Successfully Uploaded"
            title = "Prediction"

            return render_template('index.html', data=data, title=title, name=image1.filename)
    # return the data dictionary as a JSON response
    return render_template('index.html', data=data, title=title, name=name)


# if this is the main thread of execution first load the model and
# then start the server


if __name__ == "__main__":


    print(("Wait while server starts"))

    app.run(host='0.0.0.0', port=3004, debug=True)

