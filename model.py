import tensorflow as tf
import cv2
import pandas as pd
import os
import numpy as np
# from keras.models import load_model
# import kerass

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))

model = tf.keras.models.load_model(PROJECT_HOME+"/assets/ora.h5")
df=pd.read_csv(PROJECT_HOME+'/assets/dict_output.csv', sep=',',header=None)

img=cv2.imread("11.png",0)
result = np.argmax(model.predict(img))
    # alphabet = next(key for key, value in labels_values.items() if value == result) #reverse key lookup
original = db[result][2]
print("Prediction"+original)