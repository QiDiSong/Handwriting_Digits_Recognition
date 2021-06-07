import os
import mnist.model as model
import tensorflow as tf
 
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("/mnist/MNIST_data", one_hot=True)

# model 共享变量
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 784])
    # softmax函数
    y, variables = model.regression(x)

# train
y_ = tf.placeholder("float", [None, 10])
#交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#SGD考虑一部分的data，然后一部分一部分的学习，类似于mini_batch
#可以更快的学习到去往目标的路径，gd需要花费更多的时间去training达到同样的目的

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        #每次取100个实例
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #0.9192 准确率
    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False, write_state=False)
    print("Saved:", path)
