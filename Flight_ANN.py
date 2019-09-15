import numpy as np
import pandas as pd

dataset = pd.read_csv("Tweets.csv")
dataset = dataset.iloc[:,[10,1]]

X = dataset.iloc[:,0]
Y = dataset.iloc[:,1]

from nltk.tokenize import word_tokenize
from nltk.corpus import  stopwords
from nltk.stem import WordNetLemmatizer
import re 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("English"))
x = []
for tweet in X:
    tweet = re.sub('@[^\s]+','',tweet)
    words = word_tokenize(tweet)
    words = [w.lower() for w in words if w.isalpha()]
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    sentence = ' '.join(words)
    x.append(sentence)


from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1600)
x = cv.fit_transform(x).toarray()


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
ohe = OneHotEncoder(categorical_features = [0])
Y = labelencoder.fit_transform(Y)
Y = pd.get_dummies(Y)
Y = ohe.fit_transform(Y).toarray()
Y = Y[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,Y,test_size=0.20, random_state=0)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=1600, units=800, kernel_initializer="uniform"))
classifier.add(Dense(activation='relu', units = 800, kernel_initializer='uniform'))
classifier.add(Dense(activation='softmax', units=3, kernel_initializer='uniform'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, Y_train, batch_size= 100, epochs = 100)

Y_pred = classifier.predict(X_test)

for row in Y_pred:
    if (row[0] > row[1] and row[0] > row[2]):
        row[0] = 1
        row[1] = row[2] = 0
    elif (row[1] > row[0] and row[1] > row[2]):
        row[1] = 1
        row[0] = row[2] = 0
    else:
        row[2] = 1
        row[0] = row[1] = 0

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))





