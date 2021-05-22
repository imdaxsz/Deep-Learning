import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [0,1,2]
y_data = [1,3,2]

x_data = np.array(x_data)
y_data = np.array(y_data)

learning_rate = 0.1

a0 = 0.1
b0 = 0.0

x = tf.placeholder(dtype=tf.float32, shape=(3,))
y = tf.placeholder(dtype=tf.float32, shape=(3,))

a = tf.Variable(a0)
b = tf.Variable(b0)

yhat = tf.multiply(a,x) + b
loss = tf.reduce_mean(tf.square(y-yhat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate) #경사하강법
train = optimizer.minimize(loss) #gradient 계산 및 assign_sub 해줌

'''
# gradient 직접 계산
da = -(y-yhat)*x
da = tf.reduce_mean(da)
db = -(y-yhat)*1
db = tf.reduce_mean(db)
'''
#da = tf.gradients(loss, a)
#db = tf.gradients(loss, b)
#da, db = tf.gradients(loss, [a, b]) # gradient 자동계산

#a = a - learning_rate * da
#ua = tf.assign_sub(a, learning_rate*da)
#ub = tf.assign_sub(b, learning_rate*db)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(150):
    _, loss_, ua_, ub_ = sess.run([train, loss, a, b], feed_dict={x: x_data, y: y_data})
    print(loss_, ua_, ub_)
    
sess.close()
