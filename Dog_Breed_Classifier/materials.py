import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from keras.utils import np_utils
from glob import glob
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
from tensorflow.keras.applications.resnet50 import ResNet50
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

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
    #import matplotlib.pyplot as plt
    #import matplotlib.image as mpimg
    #img = mpimg.imread(path)
    #plt.imshow(img)
    #plt.show()
    if face_detector(path) ==True:
        predicted_breed = RESNET50_predict_breed(path)
        predicted_breed = predicted_breed[15:]
        print("You are human\nYour resembling dog breed is:",predicted_breed)
        message = "You are human\nYour resembling dog breed is:"+str(predicted_breed)
        return message
    elif dog_detector(path) ==True:
        predicted_breed = RESNET50_predict_breed(path)
        predicted_breed = predicted_breed[15:]
        print("You are a dog\nYour breed is:",predicted_breed)
        message = "You are a dog\nYour breed is:"+str(predicted_breed)
        return message
    else:
        print("You are neither human nor dog")
        message = "You are neither human nor dog"
        return message