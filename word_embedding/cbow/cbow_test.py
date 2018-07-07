#encoding:utf-8
#website:https://github.com/ahangchen/GDLnotes/blob/master/src/rnn/cbow.py
import zipfile
import tensorflow as tf
import numpy as np
import random
import math
import collections

from matplotlib import pylab
from sklearn.manifold import TSNE
import sys
sys.path.append("./")
from img_pickle import save_obj, load_pickle
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


def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:#构建词典和词典的索引
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]#如果在词典中，构建索引
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1#统计未知词的数量
        data.append(index)
    count[0][1] = unk_count#包含了每个词的数量
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

#num_skips中定义了应该从上下文中定义多少个词
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    context_size = 2 * skip_window#上下文的大小，没包含中心词
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.float32)#有batch_size大小个样本，则有batch_size大小个label
    batchs = np.ndarray(shape=(context_size, batch_size), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]#上下文和中心词的大小
    buffer = collections.deque(maxlen=span)#每次进入上下文和中心词大小的个数
    for _ in range(span):#以固定窗口获取数据,后面获取到数据会覆盖前面的数据,类似与buffer.extend(),只是这里使用append，增加了超出data之后的判断
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # use data of batch_size to create train_data-label set of batch_size // num_skips * num_skips
    for i in range(batch_size // num_skips):#每个词都会产生num_skips个样本
        target = skip_window  # target label at the center of the buffer，中心词的未知
        for j in range(num_skips):#对于每个要生成的样本，target代表了中心词在buffer中所处的位置
            labels[i * num_skips + j, 0] = buffer[target]#保存标签
            met_target = False#是否和中心词相等
            for bj in range(context_size):
                if bj == target:
                    met_target = True
                # 这里与skipgram的输入是反的
                # skipgram作为输入的，在这里作为输出，
                if met_target:#如果是中心词，则使用中心词的下一个
                    batchs[bj, i * num_skips + j] = buffer[bj + 1]
                else:#如果不是中心词，则直接使用该词
                    batchs[bj, i * num_skips + j] = buffer[bj]
        #添加该词
        buffer.append(data[data_index])#词的映射
        data_index = (data_index + 1) % len(data)#将词移动到下一个
    # print('generate batch')
    # print(batchs)
    return batchs, labels

vocabulary_size = 50000
#data_set = None
data_set = load_pickle('text8_data.pickle')
if data_set is None:
    # load data
    url = 'http://mattmahoney.net/dc/'
    filename = "../skipgram/small_text"

    # read data
    words = getWords(read_data(filename))
    print('Data size %d' % len(words))
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.
    data_set = {
        'data': data, 'count': count, 'dictionary': dictionary, 'reverse_dictionary': reverse_dictionary,
    }
    save_obj('text8_data.pickle', data_set)
else:
    data = data_set['data']
    count = data_set['count']
    dictionary = data_set['dictionary']
    reverse_dictionary = data_set['reverse_dictionary']

# split data
data_index = 0

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    test_size = 8
    batch, labels = generate_batch(batch_size=test_size, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch.reshape(-1)])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(-1)])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

# tensor: Train a skip-gram model, word2vec
graph = tf.Graph()
#开始构建网络
with graph.as_default():
    # Input data.,这里的输入，需要开始构建，需要包含上下文
    train_dataset = tf.placeholder(tf.int32, shape=[2 * skip_window, batch_size])
    train_labels = tf.placeholder(tf.float32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, shape=[2 * skip_window, batch_size], dtype=tf.int32)

    # Variables.随机初始化，embedding_size代表embedding后向量大小
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.通过上下文，获取到embedding后的向量
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # sum up vectors on first dimensions, as context vectors，对单词上下文求和
    embed_sum = tf.reduce_sum(embed, 0)

    # Compute the softmax loss, using a sample of the negative labels each time.构建损失函数
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases,
                                   train_labels, embed_sum,
                                   num_sampled, vocabulary_size))

    # Optimizer.定义优化器，使用自适应优化器
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    #将embedding正则化
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    # sum up vectors
    valid_embeddings_sum = tf.reduce_sum(valid_embeddings, 0)
    similarity = tf.matmul(valid_embeddings_sum, tf.transpose(normalized_embeddings))

# flow
num_steps = 100001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        # print(batch_data.shape)
        # print(batch_labels.shape)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                try:
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                except:
                    err_log_str = "err log :  [i:%d,valid_examples[i]:%s]" % (i, valid_examples[i])
                    print(err_log_str)

                for k in range(top_k):
                    try:
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                        print(log)
                    except:
                        err_log_str = "err log :  [k:%d,nearest[k]:%s]" % (k, nearest[k])
                        print(err_log_str)


    final_embeddings = normalized_embeddings.eval()
    save_obj('text8_embed.pickle', final_embeddings)

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()


words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)
