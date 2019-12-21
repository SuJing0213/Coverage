import tensorflow as tf
import input_data
import numpy as np


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(np.array(mnist.test.images).shape)
print(np.array(mnist.train.images).shape)
print(np.array(mnist.validation.images).shape)

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

W1 = tf.Variable(initial_value=tf.truncated_normal(shape=[784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]))
hidden = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(initial_value=tf.truncated_normal(shape=[500, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(hidden, W2) + b2)

# loss-func
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#  optimizer
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

for i in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(correct_prediction, feed_dict={x: [batch_xs[0]], y_: [batch_ys[0]]}))
    print(sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))

saver.save(sess, "./model.ckpt")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
