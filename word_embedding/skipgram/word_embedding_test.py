#web_site:https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb
#https://blog.csdn.net/qoopqpqp/article/details/76037334
from __future__ import print_function
from six.moves import range
import tensorflow as tf
from matplotlib import pylab

from six.moves.urllib.request import urlretrieve
import os
import zipfile
import numpy as np
import random
import collections
from sklearn.manifold import TSNE

url = "http://mattmahoney.net/dc/"


def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename



def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        #读取压缩包中的第一个文件,f.namelist是压缩包中的文件列表,split中没有传入参数时，默认会使用空格作为分隔符
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data



def build_dataset(words):
  count = [['UNK', -1]]
  #只获取前vocabulary_size个，其中vocabulary_size = 50000
  #对文档中的每个词进行统计，并取出现最多的前vocabulary_size个词
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    #注意这里给的是每个字典的大小，相当于给每个词分配一个索引，该索引按顺序增加
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  #对于文档中的每个词
  for word in words:
      #如果词出现在字典中
    if word in dictionary:
      #获取词的index
      index = dictionary[word]
    else:
      #针对未出现过的词，将索引的值设置为0
      index = 0  # dictionary['UNK']
      #统计总共未知词的个数
      unk_count = unk_count + 1
    #该data中保存了文档中每个词所对应的索引
    data.append(index)
  #重新设置未知词的个数
  count[0][1] = unk_count
  #将字典翻转
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  #返回文档中每个词所对应的索引，和文档大小一致，每个词出现的个数，词典中词和索引的映射，词典中索引和词的映射
  return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, num_skips, skip_window):
  #定义一个全局变量
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels


if __name__ == '__main__':

    filename = maybe_download('text8.zip', 31344016)
    words = read_data(filename)
    #17005207
    print('Data size %d' % len(words))

    vocabulary_size = 50000

    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.
    data_index = 0