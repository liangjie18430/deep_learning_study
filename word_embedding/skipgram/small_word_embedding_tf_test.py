#test done
#web_site:https://blog.csdn.net/qoopqpqp/article/details/76037334
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#https://www.leiphone.com/news/201706/PamWKpfRFEI42McI.html
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

import re
def getWords(data):
    rule=r"([A-Za-z-]+)"
    pattern =re.compile(rule)
    words=pattern.findall(data)
    return words

def read_data(filename):
    with open(filename,"r") as f:
        #读取压缩包中的第一个文件,f.namelist是压缩包中的文件列表,split中没有传入参数时，默认会使用空格作为分隔符
        data = f.read()

    return data


filename="small_text"
# 读取压缩包中第一个文件的全部内容
vocabulary = getWords(read_data(filename))
print('Data size', len(vocabulary))
print ('vocabulary:', vocabulary[:10])

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 200
#vocabulary_size = 50000


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# data：把原文中的word转化成ID后的串
# count：[word, freq]存储的是word和word对应的出现次数
# dictionary：词到ID的对应关系
# reverse_dictionary：ID到词的对应关系
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
# batch_size:由于batch_size 代表当前批次所需要产生的样本数，而一个词会产生num_skips个样本数，会要求该参数是num_skips的倍数
# num_skips:另一个参数叫num_skips，它代表着我们从整个窗口中选取多少个不同的词作为我们的output word
#   num_skips的解释来自于如下的网址:https://www.leiphone.com/news/201706/PamWKpfRFEI42McI.html
# skip_window:取一个word周边多远的word来训练
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0#为什么batch_size一定需要整除num_skips，batch_size相当于模型每次的batch大小个数，而num_skips是针对每个词所产生的样本个数
    assert num_skips <= 2 * skip_window#
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ],获取当前词上下文环境
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):#难道是到文章末尾后重新开始构造
        data_index = 0
    buffer.extend(data[data_index:data_index + span])#以固定窗口获取数据,后面获取到数据会覆盖前面的数据
    data_index += span#每次以固定窗口获取数据后，窗口进行滑动，
    for i in range(batch_size // num_skips):#这里为什么是batch_size //num_skips，最终只会选batch_size //num_skips个词来生成样本
        #对于每个词，获取中心词
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]#跳过中心词，由于是skip-gram模型，是使用当前词预测上下文，所以当前词是输入，其他词是输出
        #对于每个词，生成num_skips个样本
        for j in range(num_skips):
            #循环找到输出词
            while target in targets_to_avoid:#获取目标的输出词，使用while一直获取，直到目标target的词不是当前的中心词
                target = random.randint(0, span - 1)#进行随机获取，只获取num_skips个组合，即针对该词只获取num_skips个样本
            targets_to_avoid.append(target)#获取[input word,output word之类的格式]
            batch[i * num_skips + j] = buffer[skip_window]#保存每次的中心词，即input，buffer中保存来每个词的下标
            labels[i * num_skips + j, 0] = buffer[target]#保存每次的target值，即output
        if data_index == len(data):#如果data_index到了文档末尾
            buffer.extend(data[:span])#将buffer设置为文档的开头
            data_index = span #重新设置data_index
        else:
            buffer.append(data[data_index])#如果没有到文档的末尾,buffer的位置前进一位
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# batch:word对应的ID
# labels:skip-gram算法中word关联的周围的两个word（skip_window=1）
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

#print (dictionary['a'], dictionary['as'], dictionary['term'])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
# 从0-99种随机选取16个数
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()
#构建网络，疑问，根据https://www.leiphone.com/news/201706/PamWKpfRFEI42McI.html，输入应该是一个onehot后的向量
with graph.as_default():


    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.#先随机产生词汇大小和embedding_size大小的
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 根据embeddings取出与输入word（train_inputs）对应的128维的向量，内部实现原理稍后再研究
        #这里相当于从embedding中取直接取第n行，第代表的第n个词的词向量，也许这也是为什么这里不采用onehot实现的原因，可以直接使用look完成从input到embedding的转换。
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable \
            (tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    # embeddings的二范数，就是向量的长度
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    # 对向量标准化成单位向量
    normalized_embeddings = embeddings / norm
    # 根据normalized_embeddings取出与输入word（valid_dataset）对应的128维的向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    # 计算valid_embeddings和normalized_embeddings的cosine（两个向量都是单位向量）
    # 估计是从normalized_embeddings中挑选和valid_embeddings最相似的word
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

    writer = tf.summary.FileWriter('./graph', graph=graph)

    writer.close()




# Step 5: Begin training.
num_steps = 40001
#开始训练
with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        #对于每一步的训练
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # 运行优化器和损失，损失run之后会得到结果
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        # 将最相似的八个词输出到屏幕
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    try:
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    except:
                        err_log_str = "err log :  [k:%d,nearest[k]:%s]"%(k,nearest[k])
                        print(err_log_str)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    # 如果维度过高（比如1024维），建议先使用PCA降维到50维左右，再使用tsne继续降到2到3维
    # 因为直接使用tsne效率比较低
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # perplexity:一般设置成30即可，这个值一般设置在5到50之间，这个值不少很重要
    # n_components:降到2维
    # init:可选random或者pca, pca相比random更稳健一些
    # n_iter:优化的最大迭代次数，至少要设置到200
    # method:可以取两个值：1.barnes_hut（默认）：运行速度快，2.exact：运行慢，但精确
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 300  # 图上显示点的个数
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    # 每个点的label，这里取的是word的名字
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
