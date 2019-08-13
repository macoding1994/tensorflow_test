import mat4py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True
gpuConfig.log_device_placement=True

class BPNN(object):

    def __init__(self,input_n,output_n):
        self.input_n = input_n
        self.output_n = output_n
        self.X = tf.placeholder(tf.float32, shape=[None, input_n])
        self.Y = tf.placeholder(tf.float32, shape=[None, output_n])
        self.x_data = []
        self.y_data = []


        self.loss = None
        self.train_run = None
        # 保存训练模型
        self.saver = tf.train.Saver()


    def add_layer(self,inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]))
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs,Weights,biases

    def train(self, epoch):
        with tf.Session(config=gpuConfig) as sess:
            sess.run(tf.initialize_all_variables())
            ckpt = tf.train.latest_checkpoint('./tmp/model/model481')
            if ckpt:
                self.saver.restore(sess, ckpt)
            for i in range(epoch):
                loss_run, _ = sess.run([self.loss, self.train_run], feed_dict={X: x_data, y: y_data})
                print("迭代%d步  损失为：%f " % (i + 1, loss_run))

    def make_data(self,path):
        ret = mat4py.loadmat(path)
        ret = ret["data"]
        for var in ret:
            self.x_data.append(var[:-1])
            self.y_data.append([var[-1]])


if __name__ == '__main__':
    bpnn = BPNN(4,1)
    bpnn.make_data("data/data.mat")
    l1 = bpnn.add_layer()