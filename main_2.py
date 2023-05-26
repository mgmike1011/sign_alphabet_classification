import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('WZUM_dataset.csv', index_col=0)

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
                   'world_landmark_20.x','world_landmark_20.y','world_landmark_20.z','handedness.score', 'letter'] #,'handedness.label'
X = df.drop(columns_to_drop, axis=1)
y = df['letter']

handedness_label_encoder = LabelEncoder()
X['handedness.label'] = handedness_label_encoder.fit_transform(X['handedness.label'])
# ---- Dodatek do label encoder
handedness_label_encoder_keys = handedness_label_encoder.classes_
handedness_label_encoder_values = handedness_label_encoder.transform(handedness_label_encoder.classes_)
handedness_label_encoder_dictionary = dict(zip(handedness_label_encoder_keys, handedness_label_encoder_values))
print(handedness_label_encoder_dictionary)
# ----

letter_encoder = LabelEncoder()
y = letter_encoder.fit_transform(y)
# ---- Dodatek do letter encoder
letter_encoder_keys = letter_encoder.classes_
letter_encoder_values = letter_encoder.transform(letter_encoder.classes_)
letter_encoder_dictionary = dict(zip(letter_encoder_keys, letter_encoder_values))
print(letter_encoder_dictionary)
# -----

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
mdl = SVC(C=1000, degree=1, gamma=0.9, kernel='poly')
mdl.fit(X_train, y_train)
score = mdl.score(X_test, y_test)
print(score)
score = mdl.score(X_train, y_train)
print(score)