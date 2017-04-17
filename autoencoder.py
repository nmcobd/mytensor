import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist_dataset(mode='supervised', one_hot=True):
    """Load the MNIST handwritten digits dataset.

    :param mode: 'supervised' or 'unsupervised' mode
    :param one_hot: whether to get one hot encoded labels
    :return: train, validation, test data:
            for (X, y) if 'supervised',
            for (X) if 'unsupervised'
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=one_hot)

    # Training set
    trX = mnist.train.images
    trY = mnist.train.labels

    # Validation set
    vlX = mnist.validation.images
    vlY = mnist.validation.labels

    # Test set
    teX = mnist.test.images
    teY = mnist.test.labels

    if mode == 'supervised':
        return trX, trY, vlX, vlY, teX, teY

    elif mode == 'unsupervised':
        return trX, vlX, teX


def gen_batches(data, batch_size):
    data = np.array(data)

    for i in range(batch_size):
        yield data[i:i + batch_size]


class AutoEncoder:
    def __init__(self, nodes_ae=[], learning_rate=0.01, lambda_reg=5e-4, nb_epoches=100, batch_size=60):

        self.nodes_ae = nodes_ae
        self.nb_ae_layers = len(self.nodes_ae)
        self.ae_w_list = []
        self.ae_b_list = []

        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.nb_epoches = nb_epoches
        self.batch_size = batch_size
    # def _create_variables(self, nb_fea):
    #     self.ae_w = []
    #     self.decode_w = []
    #     self.ae_b = []
    #     self.decode_b = []
    #     for i, value in enumerate(self.nodes_ae):
    #         seed = 10
    #         if i == 0:
    #             self.ae_w.append(tf.Variable(initial_value=tf.truncated_normal([nb_fea, value], stddev=0.1, seed=seed)))
    #             self.ae_b.append(tf.Variable(tf.zeros([value])))
    #
    #             self.decode_w.append(tf.Variable(initial_value=tf.truncated_normal([value, nb_fea], stddev=0.1, seed=seed)))
    #             self.decode_b.append(tf.Variable(tf.zeros([nb_fea])))
    #         else:
    #             self.ae_w.append(tf.Variable(initial_value=tf.truncated_normal([self.nodes_ae[i-1], value], stddev=0.1, seed=seed)))
    #             self.ae_b.append(tf.Variable(tf.zeros([value])))
    #
    #             self.decode_w.append(tf.Variable(initial_value=tf.truncated_normal([value, self.nodes_ae[i-1]], stddev=0.1, seed=seed)))
    #             self.decode_b.append(tf.Variable(tf.zeros(self.nodes_ae[i-1])))

    def _create_placeholders(self, nb_fea):
        self.x = tf.placeholder(tf.float32, shape=[None, nb_fea])
        self.y = tf.placeholder(tf.float32, shape=[None, self.nb_y])



    def encode(self, index_layer, input=None):

        if self.ae_w is []:
            print 'no weights...'
            return input
        a = self.x
        for i in range(index_layer):
            a = tf.matmul(a, self.ae_w[i]) + self.ae_b[i]

            a = tf.nn.relu(a)
        if input is None:
            return a
        else:

            output = self.sess.run(a, feed_dict={self.x: input})
            return output

    def train_layer(self, index, data, data_vl):
        self.ae_w = []
        self.ae_b = []
        nb_fea = data.shape[1]
        for i in range(0, index-1):
            self.ae_w.append(tf.Variable(initial_value=self.ae_w_list[i], trainable=False))
            self.ae_b.append(tf.Variable(initial_value=self.ae_b_list[i], trainable=False))

        if index == 1:
            nodes_pre = nb_fea

        else:
            nodes_pre = self.nodes_ae[index-2]

        nodes_next = self.nodes_ae[index - 1]
        self.ae_w.append(tf.Variable(initial_value=tf.truncated_normal([nodes_pre, nodes_next], stddev=0.1)))
        self.ae_b.append(tf.zeros([nodes_next]))

        self.decode_b = tf.Variable(tf.zeros([nodes_pre]))
        self.sess.run(tf.initialize_all_variables())
        if index == 1:
            y = self.x
        else:
            y = self.encode(index-1)
        hidden = self.encode(index)

        output = tf.matmul(hidden, tf.transpose(self.ae_w[-1])) + self.decode_b

        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(output, y)))) +\
               self.lambda_reg * (tf.nn.l2_loss(self.ae_w[-1]))

        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        pre_loss = 10000

        for i in range(self.nb_epoches):
            np.random.shuffle(data)

            batches = [x for x in gen_batches(data, self.batch_size)]
            print 'iter ...', i, '..', self.sess.run(loss, {self.x: data_vl})
            for batch in batches:

                self.sess.run(train_step, feed_dict={self.x: batch})

        self.ae_w_list.append(self.ae_w[-1].eval())
        self.ae_b_list.append(self.ae_b[-1].eval())

    def fine_tune(self, data, labels, data_vl, labels_vl):
        print 'fine_tuning...'
        hidden = self.encode(self.nb_ae_layers)
        output = tf.nn.relu(hidden)
        output = tf.nn.dropout(output, 0.5)
        output = tf.matmul(output, self.w_ft) + self.b_ft

        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(output, self.y))))
        for w in self.ae_w:
            loss += self.lambda_reg * tf.nn.l2_loss(w)
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        pre_loss = 10000
        shuff = zip(data, labels)
        for i in range(self.nb_epoches):

            np.random.shuffle(shuff)

            batches = [x for x in gen_batches(shuff, self.batch_size)]
            pp = tf.nn.softmax(output)
            pre = self.sess.run(output, {self.x: data_vl, self.y:labels_vl})
            print 'iter ...', i, '..', 'error rate is', 100 - (100 * np.sum(np.argmax(pre, 1) == np.argmax(labels_vl, 1)) / pre.shape[0])

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.sess.run(train_step, feed_dict={self.x: x_batch, self.y:y_batch})

    def fit(self, data, labels=None, data_vl=None, labels_vl=None):
        nb_fea = data.shape[1]
        self.nb_y = labels.shape[1]
        with tf.Session() as self.sess:

            self._create_placeholders(nb_fea)

            self.w_ft = tf.Variable(initial_value=tf.truncated_normal([self.nodes_ae[-1], self.nb_y], stddev=0.1))
            self.b_ft = tf.Variable(tf.zeros([self.nb_y]))


            for i in range(1, self.nb_ae_layers+1):
                print 'train_layer', i
                self.train_layer(i, data, data_vl)
            # print self.encode(self.nb_ae_layers, data)
            self.fine_tune(data, labels, data_vl, labels_vl)

    def transform(self, data):
        print 'transforming...'

        with tf.Session() as self.sess:

            self.ae_w = []
            self.ae_b = []

            for i in range(self.nb_ae_layers):
                self.ae_w.append(tf.Variable(initial_value=self.ae_w_list[i]))
                self.ae_b.append(tf.Variable(initial_value=self.ae_b_list[i]))
            self.sess.run(tf.initialize_all_variables())
            output = self.encode(self.nb_ae_layers, data)
            return output


if __name__ == '__main__':
    x_tr, y_tr, x_vl, y_vl, x_te, y_te = load_mnist_dataset(mode='supervised')

    ae = AutoEncoder(nodes_ae=[256])

    ae.fit(x_tr, y_tr, data_vl=x_vl, labels_vl=y_vl)
