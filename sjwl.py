import tensorflow as tf
import random
import numpy as np


class SJWL:
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.labels = train_labels

        self.batch_size = 128
        self.train_epoches = 100

    def transform_labels(self, labels):
        res = np.zeros(len(labels), len(set(labels)))


    def train(self):

        X = tf.placeholder(tf.float32, shape=[self.batch_size])
        y = tf.placeholder(tf.float32, shape=[self.batch_size])

        w1 = tf.Variable(tf.truncated_normal([self.batch_size, 256]))
        b1 = tf.Variable(tf.zeros(256))

        w2 = tf.Variable(tf.truncated_normal([256, 512]))
        b2 = tf.Variable(tf.zeros(512))

        w3 = tf.Variable(tf.truncated_normal([512, 10]))

        a1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)
        a2 = tf.nn.sigmoid(tf.matmul(a1, w2) + b2)
        a3 = tf.matmul(a2, w3)

        loss = tf.nn.softmax_cross_entropy_with_logits(a3, y)

        sgd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_step = sgd.minimize(loss)

        for iter in self.train_epoches:
            if iter % 10 == 0:
                print 'iter,,,', iter
            index = range(len(self.train_data))
            random.shuffle(index)
            patch_num = len(index)/self.batch_size
            for j in range(patch_num):
                patch_data = [self.train_data[p] for p in index[j*self.batch_size:(j+1)*self.batch_size]]
                patch_labels = [self.train_labels[p] for p in index[j*self.batch_size:(j+1)*self.batch_size]]