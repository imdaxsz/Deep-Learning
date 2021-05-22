import os
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import _pickle as cPickle
import matplotlib.pyplot as plt

DATA_PATH = './cifar-10-python'

random.seed(123)
tf.set_random_seed(123)
np.random.seed(123)

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin-1')
    return dict

def one_hot(labels, vals=10):
    n = len(labels)
    out = np.zeros((n, vals))
    out[range(n), labels] = 1
    return out

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
    #im = np.hstack([images[np.random.choice(n)] for i in range(size)])
    #im = np.vstack([images[np.random.choice(n)] for i in range(size)])
    plt.imshow(im)
    plt.show()
    
class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d['data'] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float)/255
        self.labels = one_hot(np.hstack([d['labels'] for d in data]), 10)
        return self
    
    def next_batch(self, batch_size):
        x = self.images[self._i: self._i+batch_size]
        y = self.labels[self._i: self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y
    
class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(['data_batch_{}'.format(i) for i in range(1, 6)]).load()
        self.test = CifarLoader(['test_batch']).load()
        
cifar = CifarDataManager()

print('number of train images : {}'.format(len(cifar.train.images)))
print('number of train labels : {}'.format(len(cifar.train.labels)))
print('number of test images : {}'.format(len(cifar.test.images)))
print('number of test labels : {}'.format(len(cifar.test.labels)))

#images = cifar.train.images
#display_cifar(images, 5)

'''
Network Architecture
'''
init_weight = tf.initializers.truncated_normal(stddev=0.1)
init_bias = tf.initializers.constant(0.1)

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
rate = tf.placeholder(tf.float32)

# 64
w1 = tf.Variable(init_weight(shape=[3,3,3,64], dtype=tf.float32))
b1 = tf.Variable(init_bias(shape=[64]), dtype=tf.float32)
conv1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')+b1)
conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 64
w2 = tf.Variable(init_weight(shape=[3,3,64,64], dtype=tf.float32))
b2 = tf.Variable(init_bias(shape=[64]), dtype=tf.float32)
conv2 = tf.nn.relu(tf.nn.conv2d(conv1_pool, w2, strides=[1,1,1,1], padding='SAME')+b2)
conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# flat
conv2_flat = tf.reshape(conv2_pool, [-1, 8*8*64])

# fully-connected: 1024
w3 = tf.Variable(init_weight(shape=[8*8*64, 1024], dtype=tf.float32))
b3 = tf.Variable(init_bias(shape=[1024]), dtype=tf.float32)
full1 = tf.nn.relu(tf.matmul(conv2_flat, w3)+b3)
full1_drop = tf.nn.dropout(full1, rate=rate) # drop-rate

#fully-connected: 10
w4 = tf.Variable(init_weight(shape=[1024, 10], dtype=tf.float32))
b4 = tf.Variable(init_bias(shape=[10]), dtype=tf.float32)
full2 = tf.matmul(full1_drop, w4)+b4

logits = full2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
train = tf.train.MomentumOptimizer(0.005, 0.9, use_nesterov=True).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
Test
'''
x_test = cifar.test.images.reshape(10, 1000, 32, 32, 3)
y_test = cifar.test.labels.reshape(10, 1000, 10)

def test(sess):
    acc = np.mean([sess.run(accuracy, feed_dict={x: x_test[i], y: y_test[i], rate: 0.0}) for i in range(10)])
    print('test accuracy: {:.4}%'.format(acc*100))

'''
Train & Test
'''
TRAIN_SIZE = 50000  
BATCH_SIZE = 100
STEPS = TRAIN_SIZE // BATCH_SIZE
EPOCH = 25


with tf.Session() as sess:
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint("./model/") is not None:
        saver.restore(sess, tf.train.latest_checkpoint("./model/"))
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(EPOCH):
        for j in range(STEPS):
            batch = cifar.train.next_batch(BATCH_SIZE)
            _, loss_, acc = sess.run([train, loss, accuracy], feed_dict={x: batch[0], y: batch[1], rate: 0.5})
            
            if j%10 == 0:
                print('epoch: {}, steps: {}, train-loss: {}, train-accuracy: {}'.format(i+1, j+1, loss_, acc))    
        test(sess)     
    saver.save(sess, "./model/model.cpkt")    
sess.close()