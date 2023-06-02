from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import f1_score
df = pd.read_csv('datasets/WZUM_dataset_v3.csv', index_col=0)

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
                   'world_landmark_20.x','world_landmark_20.y','world_landmark_20.z','handedness.score',
                   'letter','handedness.label'] #,'handedness.label'
X = df.drop(columns_to_drop, axis=1)
y = df['letter']

# letter_encoder = LabelEncoder()
# y = letter_encoder.fit_transform(y)
# with open('letter_encoder.pkl', 'wb') as file:
#     pickle.dump(letter_encoder, file)
# # ---- Dodatek do letter encoder
# letter_encoder_keys = letter_encoder.classes_
# letter_encoder_values = letter_encoder.transform(letter_encoder.classes_)
# letter_encoder_dictionary = dict(zip(letter_encoder_keys, letter_encoder_values))
# print(letter_encoder_dictionary)
# # -----
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y, shuffle=True)

# mdl_pipe = Pipeline([
#     ('standard_scaler', StandardScaler()),
#     ('classifier', SVC(C=1000, degree=1, gamma=0.9, kernel='poly'))
# ])
mdl_pipe = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('classifier', VotingClassifier([
        ('SVC_1', SVC(C=1000, degree=1, gamma=0.9, kernel='poly')),
        ('MLP', MLPClassifier(max_iter=1000)),
        ('Lin_SVC_1', LinearSVC(C=1000, dual=False, loss='squared_hinge', multi_class='ovr', penalty='l2', max_iter=10000))
    ]))
])

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

#
# mdl = SVC(C=1000, degree=1, gamma=0.9, kernel='poly')
mdl_pipe.fit(X_train, y_train)
score = mdl_pipe.score(X_test, y_test)
print(f'Mean score TEST : {score}')
score = mdl_pipe.score(X_train, y_train)
print(f'Mean score TRAIN: {score}')
print(f'F1 score: {f1_score(y_test, mdl_pipe.predict(X_test), average=None)}')
with open('Model_SVC.pkl', 'wb') as file:
    pickle.dump(mdl_pipe, file)
