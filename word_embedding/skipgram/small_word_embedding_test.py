#web_site:
# https://blog.csdn.net/jmh1996/article/details/78395758
#一个使用较少的预料库进行模拟训练

from __future__ import print_function
from six.moves import range
import tensorflow as tf
import collections
import numpy as np

vocabulary_size = 150

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

#对词进行onehot处理

def onehot(word,vocaburary):
    l = len(vocaburary)
    #构造一个1纬的向量
    vec = np.zeros(shape=[1,1],dtype=np.float32)

    index = vocaburary.index(word)
    vec[0][index] = 1.0
    return vec



if __name__ == '__main__':

    filename = "small_text"
    data = read_data(filename)
    words = getWords(data)
    print(len(words))
    print("words[:5]    ",words[:5])

    data, count, dictionary, reverse_dictionary = build_dataset(words)


    pass