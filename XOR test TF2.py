import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

np.random.seed(123)
tf.random.set_seed(123)

# train dataset
x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[1,0], [0,1], [0,1], [1,0]], dtype=np.float32)
learning_rate = 0.01

# test dataset
pt = np.arange(0,1, 0.01)
pt_x, pt_y = np.meshgrid(pt, pt)
pt_shape = pt_x.shape
pt_x = pt_x.flatten()
pt_y = pt_y.flatten()
x_test = np.vstack((pt_x, pt_y)).T
x_test = x_test.astype(np.float32)
y_test = np.zeros(shape=(x_test.shape[0],2))
y_test = y_test.astype(np.float32)

fw = tf.keras.initializers.TruncatedNormal()
fb = tf.keras.initializers.Zeros()

w1 = tf.Variable(fw(shape=(2,15)), dtype=tf.float32)
b1 = tf.Variable(fb(shape=(1,15)), dtype=tf.float32)
w2 = tf.Variable(fw(shape=(15,2)), dtype=tf.float32)
b2 = tf.Variable(fb(shape=(1,2)), dtype=tf.float32)

# 최적화 기법 선택
optimizer = tf.keras.optimizers.Adam(learning_rate)


# 학습
EPOCH = 1000
for i in range(EPOCH):
    with tf.GradientTape() as tape:
        # 신경망 네트워크 설계
        z1 = tf.matmul(x, w1) + b1
        a1 = tf.sigmoid(z1)
        
        z2 = tf.matmul(a1, w2) + b2
        a2 = tf.sigmoid(z2)
        
        yhat = a2
        
        # 크로스엔트로피 로스 정의
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z2)
        loss = tf.reduce_mean(loss)
        
        # 정확도 계산
        correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(yhat, axis=1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        
    w1_, b1_, w2_, b2_ = tape.gradient(loss, [w1, b1, w2, b2])
    optimizer.apply_gradients(grads_and_vars=zip([w1_, b1_, w2_, b2_], [w1, b1, w2, b2]))
        
    if (i % 100 == 0):
        print(i, loss.numpy(), accuracy.numpy())

# 테스트
with tf.GradientTape() as tape:
    z1 = tf.matmul(x_test, w1) + b1
    a1 = tf.sigmoid(z1)
    
    z2 = tf.matmul(a1, w2) + b2
    a2 = tf.sigmoid(z2)
    
    yhat = a2
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test, logits=z2)
    loss = tf.reduce_mean(loss)
    
    correct = tf.equal(tf.argmax(y_test, axis=1), tf.argmax(yhat, axis=1))
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct)
    
w1_, b1_, w2_, b2_ = tape.gradient(loss, [w1, b1, w2, b2])
optimizer.apply_gradients(grads_and_vars=zip([w1_, b1_, w2_, b2_], [w1, b1, w2, b2]))

cost = loss
y_ = yhat
labels = np.argmax(y_, axis=1).astype(np.int32)

# 결과 출력
pt_x = pt_x.reshape(pt_shape)
pt_y = pt_y.reshape(pt_shape)
labels = labels.reshape(pt_shape)
plt.contourf(pt_x, pt_y, labels, cmap=plt.cm.Spectral)
plt.show()

