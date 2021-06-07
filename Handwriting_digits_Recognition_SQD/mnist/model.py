import tensorflow as tf

# Softmax Regression Model
def regression(x):
    # 产生尺寸为shape的张量(tensor)
    W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32)
    b = tf.Variable(tf.zeros([10]), dtype=tf.float32)
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y, [W, b]


# Multilayer Convolutional Network
def convolutional(x, keep_prob):
    #卷积函数
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    #池化函数
    def max_pool_2x2(x):
        # 而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，
        # 因此采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层（池化层）。
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #权重W生成函数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)    #加些噪点，看起来更像是真实数据
        return tf.Variable(initial)

    #偏置b生成函数
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # First Convolution Layer
    # 卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap（经验值）
    x_image = tf.reshape(x, [-1, 28, 28, 1])                    # input size:28*28*1   1表示黑白
    W_conv1 = weight_variable([5, 5, 1, 32])                    #patch 5*5 in_size:1,out_size:32     通道数，核个数
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # output size:28*28*32
    h_pool1 = max_pool_2x2(h_conv1)                             # output size:14*14*32


    # Second Convolutional Layer 加厚一层，图片大小14*14
    # 为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征
    W_conv2 = weight_variable([5, 5, 32, 64])                   #patch 5*5 in_size:32,out_size:64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    #output size:14*14*64
    h_pool2 = max_pool_2x2(h_conv2)                             #output size:7*7*64


    # Densely Connected Layer 加厚一层，图片大小7*7
    # 现在，图片降维到7×7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
    # 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，使用ReLU激活
    W_fc1 = weight_variable([7 * 7 * 64, 1024])                 #in_size 7*7*64,out_size:1024 特征经验 2的n次方，如32,64,1024
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])            #flat,reshape,将三维数据转为一维数据.[nsamples,7,7,64]-->[nsamples,7*7*64]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  #outsize:1024

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    # 最后，我们添加一个softmax层，就像前面的单层softmax regression一样
    W_fc2 = weight_variable([1024, 10])                         #in_size:1024,out_size:10
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)     #y:prediction

    return y, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
