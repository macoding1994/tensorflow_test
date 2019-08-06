import tensorflow as tf


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

def wind_speed_prediction():
    # 1.数据写入
    x_data = []
    y_data = []
    with open('data/data.txt', 'r') as f:
        for var in f.readlines():
            tem = []
            for i in var.split()[:-1]:
                tem.append(float(i))
            x_data.append(tem)
            y_data.append([float(var.split()[-1])])
    # 2.前向传播
    # 定义网络输入特征
    with tf.variable_scope("data"):
        # 输入特征 x [None,4]
        X = tf.placeholder(tf.float32, [None, 4], name="x_data")
        # 输出值 y float
        y = tf.placeholder(tf.float32, shape=[None,1], name="y_data")
    # 定义隐藏层：神经元8个
    with tf.variable_scope("hidden"):
        # 初始化权重和偏置
        weight_hid = tf.Variable(tf.random_normal([4, 8], mean=0.0, stddev=1.0), name="weight_hid")
        bias_hid = tf.Variable(tf.random_normal([1,8], mean=0.0, stddev=1.0), name="bias_hid")
        # 隐藏层前向传播计算
        A1 = tf.matmul(X, weight_hid) + bias_hid

    # 定义输出层：
    with tf.variable_scope("fc"):
        # 初始化权重和偏置
        weight_fc = tf.Variable(tf.random_normal([8, 1], mean=0.0, stddev=1.0), name="weight_fc")
        bias_fc = tf.Variable(tf.random_normal([1], mean=0.0, stddev=1.0), name="bias_fc")
        # 输出层前向传播计算
        y_predict = tf.matmul(A1, weight_fc) + bias_fc

    # 3.计算损失
    with tf.variable_scope("computer_loss"):
        loss = tf.reduce_mean(tf.square(y_predict - y))


    # 4.反向传播
    with tf.variable_scope("computer_loss"):
        train_run = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 运行训练网络
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for i in range(100):
            loss_run, _ = sess.run([loss, train_run], feed_dict={X: x_data, y: y_data})
            print("迭代%d步  损失为：%f" % (i, loss_run))


if __name__ == '__main__':
    wind_speed_prediction()
