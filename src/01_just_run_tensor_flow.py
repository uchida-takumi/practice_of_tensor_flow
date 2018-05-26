#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THIS tutorial from:
    https://medium.com/@curiousily/tensorflow-for-hackers-part-i-basics-2c46bc99c930
"""
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

print(tf.__version__)


v1 = tf.Variable(0.0)
p1 = tf.placeholder(tf.float32)
new_val = tf.add(v1, p1)
update = tf.assign(v1, new_val)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(5):
        sess.run(update, feed_dict={p1: 1.0})
        print('for', _, ':', sess.run(v1))
    print(sess.run(v1))


X = np.random.rand(100).astype(np.float32)
a, b = 50, 40
Y = a * X + b

plt.plot(X, Y)

# 以下はランドム項をYに加えているだけ。これを学習対象の目的変数とする。
Y = np.vectorize(lambda y: y+ np.random.normal(loc=0.0, scale=0.05))(Y)
plt.plot(X, Y)

############################
# 変数をセットし、モデルを定義し、学習を行う。
## 推定対象のパラメータをtf.Variableで定義
a_var = tf.Variable(1.0)
b_var = tf.Variable(1.0)
## 目的変数y_varはモデルで定義
y_var = a_var * X + b_var
## 最適化に利用するloss関数は以下のように定義する。
loss = tf.reduce_mean(tf.square(y_var - Y))
## 学習（lossの最小化）を実行する。
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
### 300step の学習を行う。
TRAINING_STEPS = 300
results = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(TRAINING_STEPS):
        results.append(sess.run([train, a_var, b_var]))
### 結果の確認を行う
final_result = results[-1]
a_hat = final_result[1]
b_hat = final_result[2]
y_hat = a_hat * X + b_hat
print("a:", a_hat, "b:", b_hat)

### 予測した結果を確認する。
plt.plot(X, Y);
plt.plot(X, y_hat);


