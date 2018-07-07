#encoding:utf-8
from keras.models import Sequential
from keras.layers import LSTM,Dense,Activation,SimpleRNN,Embedding
import numpy as np
from keras.preprocessing import sequence
import matplotlib.pyplot as plt


from keras.datasets import imdb

max_features=10000
maxlen=500
batch_size =2

(input_train,y_train),(input_test,y_test) = imdb.load_data(num_words=max_features)

print(len(input_train),"train_sequences")
print(len(input_test))

input_train = sequence.pad_sequences(input_train,maxlen=maxlen)
input_test = sequence.pad_sequences(input_test,maxlen=maxlen)
print(input_train.shape)

import pandas as pd

pd.read_csv()