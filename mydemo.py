
# @File    : demo.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from myBPNN import BPNN


def Load_Voice_Data(path):
    """
    这是导入数据的函数
    :param path: 数据文件的路径
    :return: 数据集
    """
    data = []
    label = []
    with open(path) as f:
        for var in f.readlines():
            ret = var.strip().split("\t")
            ret_list = []
            for i in ret:
                ret_list.append(float(i))
            data.append(ret_list[1:])
            label.append(ret_list[0])
    return data, label


def run_main():
    """
       这是主函数
    """
    # 导入数据
    path = './voice_data.txt'
    Data, Label = Load_Voice_Data(path)
    print(Data)

    # 分割数据集,并对数据集进行标准化
    Train_Data, Test_Data, Train_Label, Test_Label = train_test_split(Data, Label, test_size=1 / 4, random_state=10)
    Train_Data = Normalizer().fit_transform(Train_Data)
    Test_Data = Normalizer().fit_transform(Test_Data)

    # 设置网络参数
    # input_n = np.shape(Data)[1]
    # output_n = 1
    # hidden_n = int(np.sqrt(input_n * output_n))
    lambd = 0.001
    batch_size = 64
    learn_rate = 0.01
    epoch = 5000   #原设为1000
    iteration = 1000   #原设为10000

    # # 训练并测试网络
    # bpnn = BPNN(input_n, hidden_n, output_n, lambd)
    # train_loss, test_loss, test_accuracy = bpnn.train_test(Train_Data, Train_Label, Test_Data, Test_Label, learn_rate,
    #                                                        epoch)
    # # bpnn.save_model()
    # # 解决画图是的中文乱码问题
    # mpl.rcParams['font.sans-serif'] = [u'simHei']
    # mpl.rcParams['axes.unicode_minus'] = False
    #
    # # 结果可视化
    # col = ['Train_Loss', 'Test_Loss']
    # epoch = np.arange(epoch)
    # plt.plot(epoch, train_loss, 'r')
    # plt.plot(epoch, test_loss, 'b-.')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.legend(labels=col, loc='best')
    # plt.savefig('./训练与测试损失.png')
    # plt.show()
    # plt.close()
    #
    # plt.plot(epoch, test_accuracy, 'r')
    # plt.xlabel('Epoch')
    # plt.ylabel('Test Accuracy')
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.savefig('./测试精度.png')
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    run_main()
