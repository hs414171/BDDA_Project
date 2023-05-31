import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

x = input("Enter Comment: ")

def clean(x,voc_size = 5000,sent_length=159):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]',' ',x)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    onehot_repr = [one_hot(review,voc_size)]
    embedded_x = pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    return embedded_x
x_clean = clean(x)


model = tf.keras.models.load_model('ytnn.h5')
def predict(model,x):
    pred = model.predict(x)
    if pred > 0.4:
        print("The Comment is not spam")
    else:
        print("The Comment is spam")
    return pred
print(model.predict(x_clean))

predict(model,x_clean)

