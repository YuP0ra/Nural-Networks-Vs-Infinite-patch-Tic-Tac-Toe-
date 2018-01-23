import tensorflow as tf
import numpy as np


x, y = tf.placeholder(tf.float64, [None, 1]), tf.placeholder(tf.float64, [None, 1])


def predict(input_data):

    # Network layout will be like 1, 2, 1
    hidden_layer_neurones = 2

    w1 = tf.Variable(tf.random_normal([1, hidden_layer_neurones], dtype=tf.float32), dtype=tf.float32)
    b1 = tf.Variable(tf.random_normal([hidden_layer_neurones], dtype=tf.float32), dtype=tf.float32)
    layer1 = tf.nn.relu(tf.matmul(input_data, w1) + b1)

    w2 = tf.Variable(tf.random_normal([hidden_layer_neurones, 1], dtype=tf.float32), dtype=tf.float32)
    b2 = tf.Variable(tf.random_normal([1], dtype=tf.float32), dtype=tf.float32)
    output_layer = tf.matmul(layer1, w2) + b2

    return output_layer


def train_network(data, labels):
    prediction = predict(data)
    square_error_cost = tf.losses.mean_squared_error(labels=labels, predictions=prediction)

    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(square_error_cost)
    optimized = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            _, o = sess.run([optimizer, square_error_cost], feed_dict={x: data, y: labels})

            if i % 100 == 0:
                print("LOOP:  ", i, " is Done. Cost :", o)

        optimized.save(sess, save_path='/tmp/model.ckpt')


def predict_optimized(data):
    prediction = predict(data)
    optimized = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        optimized.restore(sess, save_path='/tmp/model.ckpt')
        out_data = sess.run(prediction, feed_dict={x: data})

    return out_data


my_data = np.arange(100, dtype=np.float32).reshape(100, 1)
my_labels = 9 * my_data + 15

# train_network(my_data, my_labels)

belal = predict_optimized([[300.]])
print(belal)