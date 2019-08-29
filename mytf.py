import tensorflow as tf
import mat4py
import numpy as np
import matplotlib.pyplot as plt


# 确定网络结构以及形状
# 第一层参数：输入：x [None, 784] 权重：[784, 64] 偏置[64],输出[None, 64]
# 第二层参数：输入：[None, 64] 权重：[64, 10] 偏置[10]，输出[None, 10]
# 流程：
# 获取数据
# 前向传播：网络结构定义
# 损失计算
# 反向传播：梯度下降优化
# 功能完善
# 准确率计算
# 添加Tensorboard观察变量、损失变化
# 训练模型保存、模型存在加载模型进行预测

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True
gpuConfig.log_device_placement=True

# train_test = True
train_test = False
mix_loss = None
count_loss = 0


def wind_speed_prediction():
    global mix_loss
    global count_loss
    # 1.数据写入
    x_data = []
    y_data = []
    # with open('data/data.txt', 'r') as f:
    #     for var in f.readlines():
    #         tem = []
    #         for i in var.split()[:-1]:
    #             tem.append(float(i))
    #         x_data.append(tem)
    #         y_data.append([float(var.split()[-1])])
    ret = mat4py.loadmat("data/data.mat")
    ret = ret["data"]
    for var in ret:
        x_data.append(var[:-1])
        y_data.append([var[-1]])

    # with open('real_data.txt','w') as f:
    #     for var in y_data:
    #         f.write(str(var[0]))
    #         f.write('\n')

    # 2.前向传播
    # 定义网络输入特征
    with tf.variable_scope("data"):
        # 输入特征 x [None,4]
        X = tf.placeholder(tf.float32, [None, 4], name="x_data")
        # 输出值 y float
        y = tf.placeholder(tf.float32, shape=[None, 1], name="y_data")
    # 定义隐藏层：神经元8个
    with tf.variable_scope("hidden1"):
        # 初始化权重和偏置
        weight_hid1 = tf.Variable(tf.random_normal([4, 8], mean=0.0, stddev=1.0, dtype=tf.float32), name="weight_hid")
        bias_hid1 = tf.Variable(tf.random_normal([1, 8], mean=0.0, stddev=1.0, dtype=tf.float32), name="bias_hid")
        # 隐藏层前向传播计算
        A1 = tf.nn.relu(tf.matmul(X, weight_hid1) + bias_hid1)

    # 定义隐藏层：神经元10个
    with tf.variable_scope("hidden2"):
        # 初始化权重和偏置
        weight_hid2 = tf.Variable(tf.random_normal([8, 10], mean=0.0, stddev=1.0, dtype=tf.float32), name="weight_hid")
        bias_hid2 = tf.Variable(tf.random_normal([1, 10], mean=0.0, stddev=1.0, dtype=tf.float32), name="bias_hid")
        # 隐藏层前向传播计算
        A2 = tf.nn.relu(tf.matmul(A1, weight_hid2) + bias_hid2)

    # 定义隐藏层：神经元6个
    with tf.variable_scope("hidden2"):
        # 初始化权重和偏置
        weight_hid3 = tf.Variable(tf.random_normal([10, 6], mean=0.0, stddev=1.0, dtype=tf.float32), name="weight_hid")
        bias_hid3 = tf.Variable(tf.random_normal([1, 6], mean=0.0, stddev=1.0, dtype=tf.float32), name="bias_hid")
        # 隐藏层前向传播计算
        A3 = tf.nn.relu(tf.matmul(A2, weight_hid3) + bias_hid3)

    # 定义输出层：
    with tf.variable_scope("fc"):
        # 初始化权重和偏置
        weight_fc = tf.Variable(tf.random_normal([6, 1], mean=0.0, stddev=1.0, dtype=tf.float32), name="weight_fc")
        bias_fc = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, dtype=tf.float32), name="bias_fc")
        # 输出层前向传播计算
        y_predict = tf.matmul(A3, weight_fc) + bias_fc

    # 3.计算损失
    with tf.variable_scope("computer_loss"):
        loss = tf.reduce_mean(tf.square(y_predict - y))

    # 4.反向传播
    with tf.variable_scope("computer_loss"):
        train_run = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 运行训练网络
    loss_list = []
    # 保存模型
    saver = tf.train.Saver()
    with tf.Session(config=gpuConfig) as sess:
        sess.run(tf.initialize_all_variables())
        # 参数加载
        ckpt = tf.train.latest_checkpoint('./tmp/model/model481061')
        if ckpt:
            saver.restore(sess, ckpt)
        if not train_test:
            for i in range(500000):
                loss_run, _ = sess.run([loss, train_run], feed_dict={X: x_data, y: y_data})
                if not mix_loss :
                    mix_loss = loss_run
                # wh, bh = sess.run([weight_hid, bias_hid])
                loss_list.append(loss_run)
                print("迭代%d步  损失为：%f 本次训练当前最小损失为： %f" % (i + 1, loss_run,mix_loss))
                # print(np.shape(wh),np.shape(bh))
                if mix_loss > loss_run:
                    mix_loss = loss_run
                    print('........%f'%mix_loss)
                    count_loss += 1
                    if count_loss > 2:
                        count_loss = 0
                        saver.save(sess,'./tmp/model/model481061/fc_nn_model')
                        print('已更新模型')
        else:
            test_result = sess.run(y_predict, feed_dict={X: x_data})
            # with open('test_data.txt','w') as f:
            #     for var in test_result:
            #         f.write(str(var[0]))
            #         f.write('\n')
            y_predict_list = []
            for var in test_result:
                y_predict_list.append(var[0])
            y_list = []
            for var in y_data:
                y_list.append(var[0])

    plt.figure()
    # plt.plot(loss_list)

    plt.scatter(range(len(y_list)),y_list,c='r')
    plt.scatter(range(len(y_predict_list)),y_predict_list,c='b',alpha=0.5)


    plt.show()


if __name__ == '__main__':
    wind_speed_prediction()
