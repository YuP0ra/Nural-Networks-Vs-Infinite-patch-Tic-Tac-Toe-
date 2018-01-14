import tensorflow as tf


class NeuralNetworkTwoHiddenLayers:
    def __init__(self, shape, name):
        self.__name = name
        self.__shape = shape
        self.__ready = False

        self.__x = tf.placeholder(tf.float32, [None, shape[0]])
        self.__y = tf.placeholder(tf.float32, [None, shape[-1]])
        self.__w1, self.__b1, self.__w2, self.__b2, self.__w3, self.__b3 = None, None, None, None, None, None

    def __initialize_network_variables(self):
        self.__w1 = tf.Variable(tf.random_normal([1, self.__shape[0]]))
        self.__b1 = tf.Variable(tf.random_normal([self.__shape[0]]))

        self.__w2 = tf.Variable(tf.random_normal([self.__shape[0], self.__shape[1]]))
        self.__b2 = tf.Variable(tf.random_normal([self.__shape[1]]))

        self.__w3 = tf.Variable(tf.random_normal([self.__shape[1], self.__shape[2]]))
        self.__b3 = tf.Variable(tf.random_normal([self.__shape[2]]))

        self.__ready = True

    def __predict(self, input_data):
        layer1_out = tf.nn.relu(tf.matmul(input_data, self.__w1) + self.__b1)
        layer2_out = tf.nn.relu(tf.matmul(layer1_out, self.__w2) + self.__b2)
        layer3_out = tf.nn.softmax(tf.matmul(layer2_out, self.__w2) + self.__b2)
        return layer3_out

    def train_network(self, data, labels):
        if self.__ready is False:
            self.__initialize_network_variables()

        prediction = self.__predict(data)
        square_error_cost = tf.losses.mean_squared_error(labels=labels, predictions=prediction)

        optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(square_error_cost)
        optimized = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                _, o = sess.run([optimizer, square_error_cost], feed_dict={self.__x: data, self.__y: labels})

                if i % 100 == 0:
                    print("LOOP:  ", i, " is Done. Cost :", o)

            optimized.save(sess, save_path='/tmp/' + str(self.__name))

    def predict_optimized(self, data):
        if self.__ready is False:
            self.__initialize_network_variables()

        prediction = self.__predict(data)
        optimized = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            optimized.restore(sess, save_path='/tmp/' + str(self.__name))
            out_data = sess.run(prediction, feed_dict={self.__x: data})

        return out_data
