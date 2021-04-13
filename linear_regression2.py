import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import keras

x_data = [[0],[1],[2]]
y_data = [[1],[3],[2]]

x_data = np.array(x_data)
y_data = np.array(y_data)

learning_rate = 0.1

#a0 = [[0.1]]
#b0 = [[0.0]]
#a0 = tf.truncated_normal(shape=(1,1), mean=0, stddev=0.1)
#b0 = tf.zeros(shape=(1,1))


a0 = tf.keras.initializers.lecun_normal()
b0 = tf.zeros(shape=(1,1))

x = tf.placeholder(dtype=tf.float32, shape=(3,1))
y = tf.placeholder(dtype=tf.float32, shape=(3,1))

a = tf.Variable(a0(shape=(1,1), dtype=tf.float32))
b = tf.Variable(b0)

yhat = tf.matmul(x, a) + b
loss = tf.reduce_mean(tf.square(y-yhat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(150):
    _, loss_, a_, b_ = sess.run([train, loss, a, b], feed_dict={x: x_data, y: y_data})
    a_ = np.squeeze(a_)
    b_ = np.squeeze(b_)
    
    print('{0:3d} {1:.5f} {2:.3f} {3:.3f}'.format(step, loss_, a_, b_))
    
sess.close()
