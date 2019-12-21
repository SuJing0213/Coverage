import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = tf.Variable(initial_value=tf.truncated_normal(shape=[3, 3, 1, 16], stddev=0.1))
b_conv1 = tf.Variable(initial_value=tf.truncated_normal(shape=[16], stddev=0.1))
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
layer1 = tf.nn.relu(h_conv1 + b_conv1)
layer2 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

W_conv2 = tf.Variable(initial_value=tf.truncated_normal(shape=[3, 3, 16, 32], stddev=0.1))
b_conv2 = tf.Variable(initial_value=tf.truncated_normal(shape=[32], stddev=0.1))
h_conv2 = tf.nn.conv2d(layer2, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
layer3 = tf.nn.relu(h_conv2 + b_conv2)
layer4 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="SAME")

layer4_flat = tf.reshape(layer4, [-1, 7*7*32])

W_fc1 = tf.Variable(initial_value=tf.truncated_normal(shape=[7*7*32, 128], stddev=0.1))
b_fc1 = tf.Variable(initial_value=tf.truncated_normal(shape=[128], stddev=0.1))
layer5 = tf.nn.relu(tf.matmul(layer4_flat, W_fc1) + b_fc1)

W_fc2 = tf.Variable(initial_value=tf.truncated_normal(shape=[128, 10], stddev=0.1))
b_fc2 = tf.Variable(initial_value=tf.truncated_normal(shape=[10], stddev=0.1))
y = tf.nn.softmax(tf.matmul(layer5, W_fc2) + b_fc2)

# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(20000):
  batch_x, batch_y = mnist.train.next_batch(50)
  if i % 100 == 0:
    # sess.run(train_step, feed_dict={x: batch_x, y_:batch_y})
    train_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels} )
    print ("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch_x, y_: batch_y})

# print ("test accuracy %g"% sess.run(accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# sess.run(tf.initialize_all_variables())
#
# y = tf.nn.softmax(tf.matmul(x,W) + b)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
# for i in range(1000):
#   batch = mnist.train.next_batch(50)
#   train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#   correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#   print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))