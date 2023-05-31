from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import sys

#
# Read input paths
#
data_path = sys.argv[1]
# data_path = '/home/milosz/RiSA_1/WZUM/sign_alphabet_classification/WZUM_dataset.csv' #Debug
output_path = sys.argv[2]
# output_path = '/home/milosz/RiSA_1/WZUM/sign_alphabet_classification/output.txt' #Debug
#
# Read data
#
data = pd.read_csv(data_path, index_col=0)
columns_to_drop = ['world_landmark_0.x','world_landmark_0.y','world_landmark_0.z','world_landmark_1.x',
                   'world_landmark_1.y','world_landmark_1.z','world_landmark_2.x','world_landmark_2.y',
                   'world_landmark_2.z','world_landmark_3.x','world_landmark_3.y','world_landmark_3.z',
                   'world_landmark_4.x','world_landmark_4.y','world_landmark_4.z','world_landmark_5.x',
                   'world_landmark_5.y','world_landmark_5.z','world_landmark_6.x','world_landmark_6.y',
                   'world_landmark_6.z','world_landmark_7.x','world_landmark_7.y','world_landmark_7.z',
                   'world_landmark_8.x','world_landmark_8.y','world_landmark_8.z','world_landmark_9.x',
                   'world_landmark_9.y','world_landmark_9.z','world_landmark_10.x','world_landmark_10.y',
                   'world_landmark_10.z','world_landmark_11.x','world_landmark_11.y','world_landmark_11.z',
                   'world_landmark_12.x','world_landmark_12.y','world_landmark_12.z','world_landmark_13.x',
                   'world_landmark_13.y','world_landmark_13.z','world_landmark_14.x','world_landmark_14.y',
                   'world_landmark_14.z','world_landmark_15.x','world_landmark_15.y','world_landmark_15.z',
                   'world_landmark_16.x','world_landmark_16.y','world_landmark_16.z','world_landmark_17.x',
                   'world_landmark_17.y','world_landmark_17.z','world_landmark_18.x','world_landmark_18.y',
                   'world_landmark_18.z','world_landmark_19.x','world_landmark_19.y','world_landmark_19.z',
                   'world_landmark_20.x','world_landmark_20.y','world_landmark_20.z','handedness.score', 'letter'] # remove 'letter'
data_to_pred = data.drop(columns_to_drop, axis=1)
#
# Encode handedness label
#
handedness_label_encoder_file = open('handedness_label_encoder.pkl', 'rb')
handedness_label_encoder = pickle.load(handedness_label_encoder_file)
data_to_pred['handedness.label'] = handedness_label_encoder.transform(data_to_pred['handedness.label'])
handedness_label_encoder_file.close()
#
# Load letter encoder
#
letter_encoder_file = open('letter_encoder.pkl', 'rb')
letter_encoder = pickle.load(letter_encoder_file)
letter_encoder_file.close()
#
# Load model and make prediction
#
with open('Model_SVC.pkl', 'rb') as file:
    mdl = pickle.load(file)
pred = mdl.predict(data_to_pred)
pred = letter_encoder.inverse_transform(pred)
with open(output_path, 'w') as file:
    for i, x in enumerate(pred):
        file.writelines(f'{x}\n')
