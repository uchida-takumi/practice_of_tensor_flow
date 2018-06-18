#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
このコードは、以下の動画を見ながらコーディングしました。
https://www.youtube.com/watch?v=yX8KuPZCAMo
"""

#############################
# --- 簡単なネットワークの実行方法
import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)

c = a * b

with tf.Session() as sess:
    output = sess.run(c)
    print(output)

#############################
# --- ネットワークの可視化
with tf.Session() as sess:
    File_Writer = tf.summary.FileWriter('abc_graph', sess.graph)
    sess.run(c)
"""
1) この後、shellで以下のコマンドを実行
tensorboard --logdir='abc_graph'
 →なんか、サーバーが起動する。

2) ブラウザで　localhost:6006 にアクセスする。
"""

#############################
# 静的なtensorであるconstantではなく、
# 外部から変数を投入できるplaceholderで処理を動かしてみる。

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

with tf.Session() as sess:
    output = sess.run(adder_node, {a:[1,3,5], b:[2,4,9]})

output.__class__


#############################
# 次にモデルを学習可能にするために、グラフに調整可能なVariableを投入します。

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([.3], tf.float32)

# Inputs and Outputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# define the model
linear_model = W * x + b

# define Loss
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Variableを全て初期化するためには、下記のように特別な呼び出しをする必要がある。
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output = sess.run([loss, W, b], {x:[1,2,3,4,5], y:[0,-1,-2,-3,-4]})
    
print(output)


#############################
# 実際に学習させるには、定義したlossを最小化するように命令を入力しなければならない。
# この最小化を実行するoptimizerを定義する。
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32, name='weight')
b = tf.Variable([.3], tf.float32, name='bias')

# Inputs and Outputs
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

# define the model
linear_model = W * x + b
linear_model = tf.identity(linear_model, name='linear_model')


# define Loss
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Variableを全て初期化するためには、下記のように特別な呼び出しをする必要がある。
init = tf.global_variables_initializer()

# optimizerを定義して、最小化命令とその対象（loss）を入れる。
optimizer = tf.train.GradientDescentOptimizer(0.01) # 0.01は learning_rate
train = optimizer.minimize(loss)

# モデルを保存するためのインスタンスをたてる。
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        feed_dict = {x:[1,2,3,4,5], y:[0,-1,-2,-3,-4]}
        output = sess.run([train, loss, W, b], feed_dict)
        
    # 学習済みのグラフをsaveしてみる。
    saver.save(sess, 'trained_model/test')
        
print(output)        


############################
# 学習した結果で予測させてみる。
import tensorflow as tf
#まず既存の学習済み状態をリセットする。
tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("trained_model/test.meta")

with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint('trained_model/'))
    
    # 変数の取得方法
    weight = sess.run('weight:0')
    bias   = sess.run('bias:0')
    
    # tensorを再取得する。
    graph = tf.get_default_graph()
    linear_model = graph.get_tensor_by_name("linear_model:0")
    x = graph.get_tensor_by_name("x:0")
    output = sess.run(linear_model, feed_dict={x:[1,2,3,4,5]})

print(output)

with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint('trained_model/'))
    print(dir(sess.graph))

