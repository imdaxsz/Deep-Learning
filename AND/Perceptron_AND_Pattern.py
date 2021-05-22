import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
x_data = [[0,0], [0,1], [1,0], [1.1]]
y_data = [[0], [0], [0], [1]]
w_data = np.random.randn(2,1)
b_data = np.zeros((4,1))

# 1단계 : 식을 세운다
x = tf.placeholder(tf.float32, shape=(4, 2)) # 텐서(Tensor) 객체
y = tf.placeholder(tf.float32, shape=(4, 1))
w = tf.placeholder(tf.float32, shape=(2, 1))
b = tf.placeholder(tf.float32, shape=(4, 1))

yhat = tf.matmul(x, w) + b
loss = 0.5 * tf.reduce_mean((y - yhat) * (y - yhat))

dw, db = tf.gradients(loss, [w, b]) # 미분을 계산해주는 객체가 제공된다.

learning_rate = 0.1

# 2단계 : 식을 계산한다
sess = tf.Session() # 계산을 담당하는 객체

# 경사하강법 적용
for i in range(100):
    loss_, dw_, db_ = sess.run([loss, dw, db],
                    feed_dict={x: x_data, y: y_data, w: w_data, b: b_data})

    w_data = w_data - learning_rate * dw_ # 경사하강법
    b_data = b_data - learning_rate * db_
    print(i, loss_)

# 학습 데이터에 대한 퍼셉트론 출력 결과
yhat_ = sess.run(yhat, feed_dict={x: x_data, y: y_data, w: w_data, b: b_data})
pred = yhat_ > 0.5

sess.close()

for i in range(4):
    print(x_data[i][0], x_data[i][1], y_data[i][0], yhat_[i], pred[i])
