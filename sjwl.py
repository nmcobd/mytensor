import tensorflow as tf
import random
import numpy as np
from sklearn.metrics import accuracy_score


class SJWL:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = self.transform_labels(train_labels)
        self.test_data = test_data
        self.test_labels = test_labels

        self.batch_size = 128
        self.train_epoches = 100

    def transform_labels(self, labels):
        res = np.zeros((len(labels), len(set(labels))))
        i = 0
        for l in labels:
            res[i][l] = 1
        return res

    def train(self):

        X = tf.placeholder(tf.float32, shape=[None, self.train_data.shape[1]])
        y = tf.placeholder(tf.float32, shape=[None, self.train_labels.shape[1]])

        w1 = tf.Variable(tf.truncated_normal([self.train_data.shape[1], 256]))
        b1 = tf.Variable(tf.zeros([256]))

        w2 = tf.Variable(tf.truncated_normal([256, 512]))
        b2 = tf.Variable(tf.zeros([512]))

        w3 = tf.Variable(tf.truncated_normal([512, 10]))
        b3 = tf.Variable(tf.zeros([10]))

        a1 = tf.nn.relu(tf.matmul(X, w1) + b1)
        a2 = tf.nn.relu(tf.matmul(a1, w2) + b2)
        a3 = tf.matmul(a2, w3) + b3
        y_ = tf.nn.softmax(a3)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=a3))

        sgd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_step = sgd.minimize(loss)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        for iter in range(self.train_epoches):
            if iter % 5 == 0:
                print 'iter,,,', iter
                pre = sess.run(y_, feed_dict={X: self.test_data})
                res = []
                for p in pre:
                    res.append(np.argmax(p))
                print accuracy_score(self.test_labels, res)
            index = range(len(self.train_data))
            random.shuffle(index)
            patch_num = len(index)/self.batch_size
            for j in range(patch_num):
                # print j
                patch_data = [self.train_data[p] for p in index[j*self.batch_size:(j+1)*self.batch_size]]
                patch_labels = [self.train_labels[p] for p in index[j*self.batch_size:(j+1)*self.batch_size]]

                feed_dict = {X: patch_data, y: patch_labels}
                sess.run(train_step, feed_dict=feed_dict)
