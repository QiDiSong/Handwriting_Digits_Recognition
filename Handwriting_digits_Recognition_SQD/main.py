import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from mnist import model

#print(tf.__version__)

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

# restore trained data
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")


with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()
    print(sess.run(y1, feed_dict={x: input}).flatten().tolist())

def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()
    print(sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist())

# webapp 导入__name__,确定资源所在的路径
app = Flask(__name__)

@app.route('/api/mnist', methods=['POST'])
def mnist():
    #标准化数据
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    #一维数组，输出10个预测概率
    output1 = regression(input)
    output2 = convolutional(input)

    return jsonify(results=[output1, output2])

#获取到数据后，把数据传入HTML模板文件中，模板引擎负责渲染HTTP响应数据，然后返回响应数据给客户端(浏览器)
@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
