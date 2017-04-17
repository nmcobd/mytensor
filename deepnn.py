from autoencoder import *


class DeepNN(AutoEncoder):

    def __init__(self, nodes_ae, nodes_fc, learning_rate=0.001, lambda_reg=0.0005, nb_epoches=300):
        AutoEncoder.__init__(self, nodes_ae, learning_rate, lambda_reg, nb_epoches)

        self.nodes_fc = nodes_fc
        self.nb_fc_layers = len(self.nodes_fc)

        self.fc_w_list = []
        self.fc_b_list = []

    def _create_fc_variables(self):
        seed = 10
        self.fc_w = []
        self.fc_b = []
        nb_fea = 784
        for index, value in enumerate(self.nodes_fc):
            if index == 0:
                self.fc_w.append(tf.Variable(initial_value=tf.truncated_normal([self.nodes_ae[-1], value], stddev=0.1, seed=seed)))
                self.fc_b.append(tf.Variable(initial_value=tf.truncated_normal([value])))

            else:
                self.fc_w.append(tf.Variable(initial_value=tf.truncated_normal([self.nodes_fc[index-1], value], stddev=0.1, seed=seed)))
                self.fc_b.append(tf.Variable(initial_value=tf.truncated_normal([value])))

    def get_fc_output(self, input, train=False):
        output = input
        for i in range(self.nb_fc_layers):

            output = tf.matmul(output, self.fc_w[i]) + self.fc_b[i]
            if i != self.nb_fc_layers - 1:
                output = tf.nn.relu(output)
            if train:
                output = tf.nn.dropout(output, 0.5)
        return output

    def train_dnn(self, data, labels):
        print 'autoencoding...'
        self.nb_y = labels.shape[1]
        self.fit(data, labels)
        print 'autoendoing done.'
        with tf.Session() as self.sess:
            self.ae_w = []
            self.ae_b = []
            for i in range(self.nb_ae_layers):
                self.ae_w.append(tf.Variable(initial_value=self.ae_w_list[i]))
                self.ae_b.append(tf.Variable(initial_value=self.ae_b_list[i]))
            self._create_fc_variables()
            ae_out = self.encode(self.nb_ae_layers)
            # ae_out = self.x
            ae_out = tf.nn.relu(ae_out)
            output = self.get_fc_output(ae_out, train=True)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, self.y))
            for w in self.ae_w:
                loss += self.lambda_reg * tf.nn.l2_loss(w)
            for w in self.fc_w:
                loss += self.lambda_reg * tf.nn.l2_loss(w)
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            self.sess.run(tf.initialize_all_variables())
            for i in range(self.nb_epoches):
                if i % 5 == 0:
                    y_ = tf.nn.softmax(output)
                    pre = self.sess.run(y_, feed_dict={self.x: data})
                    print 'iter...', i, ', acc is', float(np.sum(np.argmax(pre, 1) == np.argmax(labels, 1))) / pre.shape[0]
                self.sess.run(train_step, feed_dict={self.x: data, self.y: labels})

            for i in range(self.nb_fc_layers):
                self.fc_w_list.append(self.fc_w[i].eval())
                self.fc_b_list.append(self.fc_b[i].eval())

    def predict(self, data, labels=None):
        print 'predicting...'

        with tf.Session() as self.sess:
            self.ae_w = []
            self.ae_b = []
            self.fc_w = []
            self.fc_b = []
            for i in range(self.nb_ae_layers):
                self.ae_w.append(tf.Variable(initial_value=self.ae_w_list[i]))
                self.ae_b.append(tf.Variable(initial_value=self.ae_b_list[i]))
            for i in range(self.nb_fc_layers):
                self.fc_w.append(tf.Variable(initial_value=self.fc_w_list[i]))
                self.fc_b.append(tf.Variable(initial_value=self.fc_b_list[i]))
            self.sess.run(tf.initialize_all_variables())
            ae_out = self.encode(self.nb_ae_layers, data)
            # ae_out = self.x
            ae_out = tf.nn.relu(ae_out)
            fc_out = self.get_fc_output(ae_out)
            y_ = tf.nn.softmax(fc_out)

            pre = self.sess.run(y_, feed_dict={self.x: data})
            if labels is not None:
                print 'try testing...'
                for i in range(min(len(pre), 30)):
                    print np.argmax(labels[i]), np.argmax(pre[i])
                print 'error rate is', 100 - (100 * np.sum(np.argmax(pre, 1) == np.argmax(labels, 1)) / pre.shape[0])
            return np.argmax(pre, 1)
