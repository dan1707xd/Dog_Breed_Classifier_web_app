# Dog Identification and Breed Classifier Web App using ResNet50 as pretrained CNN
# Project Motivation/Details
This project implements an algorithm to detect whether a picture contains a human or a dog. Upon detection, it uses a CNN to classify the dog breed in case of a dog picture, otherwise, in case of a human, it returns the closest resemblance to a dog breed. The algorithm accomplishes this task as follows:
1. Using OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images, we determine if the picture is that of a human.
2. We use a pre-trained ResNet-50 model to detect dogs in images.
3. In both of the cases above, the image is then passed to the CNN model (that is a pre-trained ResNet50 CNN) to identify which dog breed is a close resemblance to the human or the dog belongs to. In this implementation, we have 133 classes for the dog breeds.

The Jupyter notebook (available in the Dog_Breed_Classifier folder both as an html or ipynb file) goes into the details of the coding work involved to accomplish our task. The notebook is delineated as follows:
1. Import Datasets
2. Detect Humans
3. Detect Dogs
4. Create a CNN to Classify Dog Breeds (from Scratch)
5. Use a CNN to Classify Dog Breeds (using Transfer Learning)
6. Create a CNN to Classify Dog Breeds (using Transfer Learning)
7. Write your Algorithm
8. Test Your Algorithm


Finally, we created a simple web app that takes a picture (user uploaded) and predicts as per our algorithm.

# Prompt to Upload a picture
![Screenshot 1](Dog_Breed_Classifier/1.jpg)
# Prediction on a Labrador Retriever
![Screenshot 2](Dog_Breed_Classifier/2.jpg)
![Screenshot 3](Dog_Breed_Classifier/3.jpg)
![Screenshot 4](Dog_Breed_Classifier/4.jpg)

# Installation
Make sure any Python 3.* is installed alongside the pandas, numpy, matplotlib, sklearn , pickle, sqlalchemy,NLTK, Plotly and Flask libraries.


# File Descriptions
1. The App folder including the templates folder and "run.py" for the web app.
2. The Data folder containing "Disaster_Data.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py".
3. The Models folder including "model_1.sav" and "train_classifier.py" for the NLP pipeline and Machine Learning model.
4. README file
5. Jupyter Notebook that contains all details and steps of the functions, transformers and analysis created.

# Instructions
Run the following in the project's root directory:
1. **python data/process_data.py** ==> to generate the database
2. **python models/train_classifier.py** ==> to generate the classifier model that will run the web app. Note: 3 models (model_1, model_2, model_3 are defined and can be used in def main()
3. **python run.py** ==> to run web app (make sure classifier model name matches the one in run.py)


# Licensing, Authors, Acknowledgements
Thanks to Figure-8 for the data!
Thanks to UDACITY, Kaggle and StackOverflow for providing insight and solution to complications encountered along the way!
