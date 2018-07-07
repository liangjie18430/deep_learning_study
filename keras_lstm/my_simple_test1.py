#encoding:utf-8
from keras.models import Sequential
from keras.layers import LSTM,Dense,Activation
import numpy as np

import matplotlib.pyplot as plt


a=[1,2,3,4,5,6,7,8,9,10]
a = np.reshape(a,[-1,1])
b=[i+2 for i in a]
b=np.reshape(b,[-1,1])

c=[i*i for i in a]
c = np.reshape(c,[-1,1])


def build_layers():
    model = Sequential()
    #从a到b的输入和输出维度都为10
    #return_sequence: Boolean.False返回在输出序列中的最后一个输出；True返回整个序列
    model.add(LSTM(input_shape=(None,1),output_dim=1,return_sequences=True))
    print(model.layers)
    #第一个参数为单元数
    #model.add(LSTM(100,return_sequences=False))
    #model.add(Dense(10))
    #model.add(Activation('softmax'))
    model.add(Activation('linear'))
    model.compile(loss='mse',optimizer='rmsprop')
    return model
def train_model(train_x, train_y, test_x, test_y):
    model=build_layers()

    model.fit(train_x,train_y,batch_size=7,epochs=1000,validation_split=0.1)
    predict=model.predict(test_x)
    predict=np.reshape(predict,(predict.size,))
    print("predict:",predict)
    print("test_y:",test_y)
    try:
        fig = plt.figure(1)
        plt.plot(predict, 'r:')
        plt.plot(test_y, 'g-')
        plt.legend(['predict', 'true'])
    except Exception as e:
        print(e)
    return predict, test_y

def reshape_dataset(train):
    """
    主要为了适配lstm的数据输入格式
    :param train:
    :return:
    """

    trainx = np.reshape(train,(train.shape[0],1,train.shape[1]))
    return trainx


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    train_x, test_x, train_y, test_y = train_test_split(a, b, test_size=0.2, random_state=0)
    train_x = reshape_dataset(train_x)
    test_x = reshape_dataset(test_x)
    train_y = reshape_dataset(train_y)
    test_y = reshape_dataset(test_y)
    print(train_x)
    print(test_x)
    print(train_y)
    print(test_y)
    train_model(train_x,train_y,test_x,test_y)