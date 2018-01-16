import tensorflow as tf


class NeuralNetworkFourHiddenLayers:
    def __init__(self, shape, name):
        self.__name = name
        self.__shape = shape
        self.__ready = False
        self.__sess = None

        self.__x = tf.placeholder(tf.float32, [None, shape[0]])
        self.__y = tf.placeholder(tf.float32, [None, shape[-1]])
        self.__w1, self.__b1, self.__w2, self.__b2, self.__w3, self.__b3 = None, None, None, None, None, None

    def __initialize_network_variables(self):
        self.__w1 = tf.Variable(tf.random_normal([self.__shape[0], self.__shape[1]], stddev=0.01))
        self.__b1 = tf.Variable(tf.random_normal([self.__shape[1]]))

        self.__w2 = tf.Variable(tf.random_normal([self.__shape[1], self.__shape[2]], stddev=0.01))
        self.__b2 = tf.Variable(tf.random_normal([self.__shape[2]]))

        self.__w3 = tf.Variable(tf.random_normal([self.__shape[2], self.__shape[3]], stddev=0.01))
        self.__b3 = tf.Variable(tf.random_normal([self.__shape[3]]))

        self.__w4 = tf.Variable(tf.random_normal([self.__shape[3], self.__shape[4]], stddev=0.01))
        self.__b4 = tf.Variable(tf.random_normal([self.__shape[4]]))

        self.__prediction = self.__predict(self.__x)
        self.__cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.__y, logits=self.__prediction)

        self.__optimizer = tf.train.AdamOptimizer().minimize(self.__cross_entropy)

        self.__start_session()

        self.__ready = True

    def __predict(self, input_data):
        layer1_out = tf.nn.relu(tf.matmul(input_data, self.__w1) + self.__b1)
        layer2_out = tf.nn.relu(tf.matmul(layer1_out, self.__w2) + self.__b2)
        layer3_out = tf.nn.relu(tf.matmul(layer2_out, self.__w3) + self.__b3)

        return tf.nn.softmax(tf.matmul(layer3_out, self.__w4) + self.__b4)

    def __start_session(self):
        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())

    def train_network(self, data, labels):
        if self.__ready is False:
            self.__initialize_network_variables()

        for i in range(100):
            _, o = self.__sess.run([self.__optimizer, self.__cross_entropy], feed_dict={self.__x: data, self.__y: labels})

        tf.train.Saver().save(sess=self.__sess, save_path='/tmp/' + str(self.__name))

    def predict_optimized(self, data):
        if self.__ready is False:
            self.__initialize_network_variables()
            tf.train.Saver().restore(self.__sess, save_path='/tmp/' + str(self.__name))

        prediction = self.__predict(data)
        out_data = self.__sess.run(prediction, feed_dict={self.__x: data})

        return out_data
