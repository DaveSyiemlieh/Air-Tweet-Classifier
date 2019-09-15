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

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 550, criterion = 'entropy')

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))





