#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################
# Building a Simple Neural Network — TensorFlow for Hackers (Part II)
## https://medium.com/@curiousily/tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b
###################################

# import 
import tensorflow as tf
import numpy as np
import pandas as pd

# set up
# load data
fi_tpx = pd.read_csv('data/fi_tpx.csv')
fi_tpx = fi_tpx.ix[1:,3:10] #適当に数値のデータだけに絞り込む
fi_tpx = pd.DataFrame(fi_tpx, dtype=np.float64)
fi_tpx[fi_tpx > 1.00000000e+299] = 0 #異常に大きな値は0に置換しておく

train_x, train_y = fi_tpx.values[:500, :-1], fi_tpx.values[:500, -1]
test_x, test_y = fi_tpx.values[500:600, :-1], fi_tpx.values[500:600, -1]

# 学習データの構造
train_x.shape
train_y.shape

# Building the Neural Network

def my_network(x, weights, biases, keep_prob):
    """
    tf.matmul(a,b)はテンソルaとbのかけ算みたいなもの。
    tf.add(a,b)はa+b
    layer_1のアクティベート関数はReLUを利用
    時々drop outさせたいので、その確率をkeep_probで定義。    
    """
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

## 上記の my_network　の学習パラメータを定義する。
n_hidden_1 = 3 #neuronの数
n_input = train_x.shape[1]
n_output = 1

weights = {
         'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]))
        ,'out': tf.Variable(tf.random_normal([n_hidden_1, n_output]))
        }
biases = {
         'b1': tf.Variable(tf.random_normal([n_hidden_1]))
        ,'out': tf.Variable(tf.random_normal([n_output]))
        }

keep_prob = tf.placeholder("float") # tf.placeholder()は定義したネットワークの外からの入力を定義するらしい

training_epochs = 500
display_step = 100
batch_size = 32

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

loss = tf.losses.mean_squared_error(y, my_network(x, weights, biases, keep_prob))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
prediction = my_network(x, weights, biases, keep_prob)

# Let's GO
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(train_x) / batch_size)
        x_batches = np.array_split(train_x, total_batch)
        y_batches = np.array_split(train_y.reshape((-1,1)), total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, c, w_h1, w_out, b_b1, b_out = sess.run(
                                    [optimizer, loss, weights['h1'], weights['out'], biases['b1'], biases['out']],
                                    feed_dict={
                                            x: batch_x,
                                            y: batch_y,
                                            keep_prob: 0.8
                                            })
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    
    print("Optimization Finished!")


# 重みなどのパラメータ情報の最終値は下記に保存されている。
print(w_h1)
print(w_out)
print(b_b1)
print(b_out)    




