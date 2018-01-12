import tensorflow as tf
import numpy as np


x, y = tf.placeholder(tf.float32, [None, 1]), tf.placeholder(tf.float32, [None, 1])


def predict(input_data):
    # Network layout will be like 1, 2, 1
    hidden_layer_neurones = 128
    layers_stddev = 1. / hidden_layer_neurones

    w1 = tf.Variable(tf.random_normal([1, hidden_layer_neurones], stddev=layers_stddev))
    b1 = tf.Variable(tf.random_normal([hidden_layer_neurones], stddev=layers_stddev))
    layer1 = tf.nn.relu(tf.matmul(input_data, w1) + b1)

    w2 = tf.Variable(tf.random_normal([hidden_layer_neurones, hidden_layer_neurones], stddev=layers_stddev))
    b2 = tf.Variable(tf.random_normal([hidden_layer_neurones], stddev=layers_stddev))
    layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

    w2 = tf.Variable(tf.random_normal([hidden_layer_neurones, 1], stddev=layers_stddev))
    b2 = tf.Variable(tf.random_normal([1], stddev=layers_stddev))
    output_layer = tf.matmul(layer2, w2) + b2

    return output_layer


def train_network(data, labels):
    prediction = predict(data)
    square_error_cost = tf.reduce_sum(tf.square(labels - prediction))

    optimizer = tf.train.AdamOptimizer().minimize(square_error_cost)
    optimized = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100000):
            _, o = sess.run([optimizer, square_error_cost], feed_dict={x: data, y: labels})
            print("LOOP: ", i, " is Done. Cost :", o)

        optimized.save(sess, save_path='/tmp/model.ckpt')


def predict_optimized(data):
    prediction = predict(data)
    optimized = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        optimized.restore(sess, save_path='/tmp/model.ckpt')
        out_data = sess.run(prediction, feed_dict={x: data})

    return out_data

    pass


def shuffle(a, b):
    seed = np.random.random_integers(10000)
    np.random.seed(seed)
    np.random.shuffle(a)
    np.random.seed(seed)
    np.random.shuffle(b)


my_data = np.arange(100, dtype=np.float32).reshape(100, 1)
my_labels = my_data * my_data

train_network(my_data, my_labels)


