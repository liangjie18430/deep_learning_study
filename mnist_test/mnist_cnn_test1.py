#encoding:utf-8
from __future__ import print_function

import tensorflow as tf


#定义读取数据

def read_mnist_data_with_tf():
    """
    读取手写数据集mnist，其中images是图像的意思
    :return:
    """
    from tensorflow.examples.tutorials.mnist import input_data

    mnist_data = input_data.read_data_sets("MNIST_data")
    print("train.shape:",mnist_data.train.images.shape)
    print("test.shape:",mnist_data.test.images.shape)
    return mnist_data



def build_nn():
    """
    构建一个cnn的神经网络
    :return:
    """
    #
    tf.layers.Conv2D()

    pass


if __name__ == '__main__':

    pass