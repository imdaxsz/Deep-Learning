import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data=np.array([[0,0],[0,1],[1,0],[1,1]])
y_data=np.array([[0],[1],[1],[0]])

x = tf.placeholder(tf.float32, shape=(None, 2))
y = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal(shape=(2,3), mean=0, stddev=1.0, dtype=tf.float32))
b1 = tf.Variable(tf.zeros(shape=(1,3), dtype=tf.float32))

w2 = tf.Variable(tf.random_normal(shape=(3,1), mean=0, stddev=1.0, dtype=tf.float32))
b2 = tf.Variable(tf.zeros(shape=(1,1), dtype=tf.float32))

z1 = tf.matmul(x, w1)+b1
a1 = tf.sigmoid(z1)

z2 = tf.matmul(a1, w2)+b2
a2 = tf.sigmoid(z2)

yhat = a2

loss = 0.5 * tf.reduce_mean((y - yhat) * (y - yhat))
learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, loss_, yhat_ = sess.run([train, loss, yhat], feed_dict={x:x_data, y:y_data})
    print(i, loss_, yhat_[0], yhat_[1], yhat_[2], yhat_[3])

sess.close()



