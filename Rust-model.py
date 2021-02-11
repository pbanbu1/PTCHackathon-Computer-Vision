import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random


DATADIR = "./test_images"
CATEGORIES = ['Rust', 'No Rust']

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to rust or no rust dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
IMG_SIZE = 75

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #path to rust or no rust dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()

random.shuffle(training_data)
Z = []
Y = []


for features, label in training_data:
    Z.append(features)
    Y.append(label)
X = np.array(Z).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
