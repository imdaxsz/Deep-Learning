import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

# 파라미터 성능평가시 고정
tf.set_random_seed(123)
np.random.seed(123)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([[0], [1], [1], [0]])
learning_rate = 0.01

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# 초기화 방법 및 변수 정의
h1_w23 = tf.truncated_normal(shape=(2,3), mean=0, stddev=1.0, dtype=tf.float32)
h1_b13 = tf.zeros(shape=(1,3))
w1 = tf.Variable(h1_w23)
b1 = tf.Variable(h1_b13)

o_w31 = tf.truncated_normal(shape=(3,1), mean=0, stddev=1.0, dtype=tf.float32)
o_b11 = tf.zeros(shape=(1,1))
w2 = tf.Variable(o_w31)
b2 = tf.Variable(o_b11)

# 신경망 네트워크 설계
z1 = tf.matmul(x, w1) + b1
a1 = tf.sigmoid(z1)

z2 = tf.matmul(a1, w2) + b2
a2 = tf.sigmoid(z2)

yhat = a2

# 크로스엔트로피 로스 정의
loss = -y * tf.log(yhat) - (1 - y) * tf.log(1 - yhat)
loss = tf.reduce_mean(loss)

# 최적화 기법 선택
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# 정확도 계산
#thresholding = tf.greater_equal(yhat, 0.5)
thresholding = yhat > 0.5
thresholding = tf.cast(thresholding, tf.float32)
prediction = tf.cast(tf.equal(thresholding, y), tf.float32)
accuracy = tf.reduce_mean(prediction)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

Loss = []
EPOCH = 300
for i in range(EPOCH):
    outs = sess.run([train, loss, yhat, accuracy], feed_dict={x: x_data, y: y_data})
    print(i, outs[1], outs[3])
    Loss.append(outs[1])

plt.plot(Loss)
plt.show()

sess.close()
