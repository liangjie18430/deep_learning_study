#web_site:https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb
#中文解释所在位置:https://www.sohu.com/a/140079448_505915
import sys
sys.path.append("./")
x = [[5, 7, 8], [6, 3], [3], [1]]
import helpers

import numpy as np
import tensorflow as tf
import helpers

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units
# [encoder_max_time, batch_size]，max_time可以理解为句子的最大长度
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)#[10,20]

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)#[8,100,20]
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)#[9,100,20]

#https://blog.csdn.net/songhk0209/article/details/71134698
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)#20个隐藏单元层

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    #time_major=True查看注解，must be `[max_time, batch_size, depth]`.
    dtype=tf.float32, time_major=True,
)

del encoder_outputs


decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,

    initial_state=encoder_final_state,

    dtype=tf.float32, time_major=True, scope="plain_decoder",
)


decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),#label.shape=[9,100,10],
    #9可以理解为每一行，label[0]代表第一行，label[1]代表第二行进行onehot，100是每一行有100列，10可以理解为词典的大小，
    #即把每个句子出现的词，按照onehot方式映射到每个词典。
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.global_variables_initializer())

batch_ = [[6], [3, 4], [9, 8, 7]]

batch_, batch_length_ = helpers.batch(batch_)
print('batch_encoded:\n' + str(batch_))

din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32),
                            max_sequence_length=4)
print('decoder inputs:\n' + str(din_))

pred_ = sess.run(decoder_prediction,
    feed_dict={
        encoder_inputs: batch_,
        decoder_inputs: din_,
    })
print('decoder predictions:\n' + str(pred_))


batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)

print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)



def next_feed():
    batch = next(batches)#batch中装入的是100个队列，可以理解为不同的句子
    encoder_inputs_, _ = helpers.batch(batch)#转换成一样长度的向量，_中记录了有多少个不为0
    decoder_targets_, _ = helpers.batch(#添加句子结束标记，此处EOS定义为1
        [(sequence) + [EOS] for sequence in batch]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,#[8,100]
        decoder_inputs: decoder_inputs_,#[9,100],以EOS开头
        decoder_targets: decoder_targets_,#[9,100]，以EOS结尾
    }

loss_track = []


max_batches = 3001
batches_in_epoch = 1000
writer = tf.summary.FileWriter('./graph', graph=sess.graph)
writer.close()
try:
    for batch in range(max_batches):
        fd = next_feed()#[8,100],8行100列，1列代表一个句子
        #查看一些变量的值
        label = sess.run(tf.one_hot(fd[decoder_targets], depth=vocab_size, dtype=tf.float32))
        #encoder_inputs[8,100],encoder_inputs_embedded_test = [8,100,20]
        encoder_inputs_embedded_test = tf.nn.embedding_lookup(embeddings, fd[encoder_inputs])





        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()

except KeyboardInterrupt:
    print('training interrupted')


import matplotlib.pyplot as plt
plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))