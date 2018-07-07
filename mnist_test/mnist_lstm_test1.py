#encoding:utf-8
#web_site:https://blog.csdn.net/jerr__y/article/details/61195257
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug


def print_tf():
    pass


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#导入数据,onehot后将二维的数据，转换成了一维
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

print("train.images.shape:",mnist.train.images.shape)
print("test.images.shape:",mnist.test.images.shape)

sess_print = tf.Session()
#准备构建一个rnn网络
#设置学习率
lr=1e-3
#设置batch_size为占位符，以便使用不同的batch_size,
#不指定shape代码是一个证书
batch_size = tf.placeholder(tf.int32,[])

#由于使用lstm,以第一行作为最开始的输入

input_size = 28

timestep_size = 28

#指定了每个隐藏层节点的个数
hidden_units = 256

#LSTM的层数

layer_num = 2

#最后输出分类类别数据，
class_num = 10

_X = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,[None,class_num])
#dropout层keep_prob的大小
keep_prob = tf.placeholder(tf.float32,shape=[])

#开始构建rnn的输入,shape = (batch_size, timestep_size, input_size)

X = tf.reshape(_X,[-1,28,28])
"""
#定义一个lstm_层,内置激活函数是tanh,需要设置隐藏层单元数
#隐藏单元个数256
lstm_cell = rnn.BasicLSTMCell(num_units=hidden_units, forget_bias=1.0)
#添加dropoutlayer层，一般只设置output_keep_prob
lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prob)

#实现多层,注意乘号和中括号,中括号代表了list
mlstm_cell = rnn.MultiRNNCell([lstm_cell]*layer_num)
"""

stacked_rnn = []
for iiLyr in range(layer_num):
    stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(num_units=hidden_units, state_is_tuple=True))
mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
#使用全0初始化，这里的state什么意思
#这里直接对构建的多个层进行初始化
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)


writer_print=tf.summary.FileWriter("./printlog",sess_print.graph)

writer_print.close()
outputs = list()

state = init_state

with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        #timestep从0开始，过滤掉timestep=0
        if timestep>0:
            #共享变量,可以尝试删除该变量后的效果。
            tf.get_variable_scope().reuse_variables()
        #此时x是一个[1,28,18]的图像输入，这里调用了MultiRNNCell中的call方法
        #抛出异常with input shapes: [?,512], [284,1024].
        (cell_output,state) = mlstm_cell(X[:,timestep,:],state)
        #尝试能否在次进行state的shape的输出，按照公式的推断，应该是256的shape
        #这里为什么需要保存所有单元的输出
        outputs.append(cell_output)

#这里直接是最后步骤的输出，-1相当于取队列的最后一个值
h_state = outputs[-1]
#定义softmax的连接权重和偏置,由于是定义softmax的相关函数，shape可以确定，
#使用截断高斯函数初始化

W = tf.Variable(tf.truncated_normal([hidden_units, class_num], stddev=0.1), expected_shape=[hidden_units, class_num], dtype=tf.float32)

bias = tf.Variable(expected_shape=[class_num],dtype=tf.float32,initial_value=tf.constant(0.1,shape=[class_num])
                )

#此时w连接的是输出层

y_pred = tf.nn.softmax(tf.matmul(h_state,W)+bias)

# 损失函数和评估函数
# 采用logloss损失
cross_entropy = -tf.reduce_mean(y*tf.log(y_pred))
#采用adam优化器
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#由于是一个多分类问题，第二个参数1代表axis=1，查看下标是否相等
correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
#使用session运行
#



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./mnist_test1_log', graph=sess.graph)

    #训练2000次
    for i in range(2000):
        _batch_size = 128
        #从数据流中获取相关的数据，batch shape(128,784)
        batch = mnist.train.next_batch(_batch_size)
        #shape操作是一个Tensor，需要运行后才能有结果
        #print("shape of batch :", sess.run(tf.shape(batch)))
        #每隔200次就输出精确度
        if (i+1)%200 == 0:
            #accuracy带入后包含h_state,W,b,W和b都有初始化函数进行初始化
            #h_state中包含X,而X包含_X,需要对_X进行输入
            #同时还需要的参数有batch_size,keep_proba,y的真实值
            #即最开始定义的几个参数
            train_accuracy  = sess.run(accuracy, feed_dict={_X:batch[0], y:batch[1], keep_prob:0.5, batch_size:_batch_size})
            print("Iter %d,step %d,training accuracy %g"%(mnist.train.epochs_completed,i+1,train_accuracy))
        #debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        #debug_sess.run(train_op, feed_dict={_X:batch[0], y:batch[1], keep_prob:0.5, batch_size:_batch_size})

        sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

    # 计算测试数据的准确率,注意keep_prob的变化，从0.5变化到1.0
    print("test accuracy %g" % sess.run(accuracy, feed_dict={
        _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size: mnist.test.images.shape[0]}))


writer.close()


sess_print.close()














