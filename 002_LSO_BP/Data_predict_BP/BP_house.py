import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from logger import logger
from CLSO_bp_main import mse, rmse, mae, TV, TSD
import pdb


def BP(data, learning_rate = 0.005, max_epoch=1000):
    # 1-1 加载数据
    boston = data
    pdb.set_trace()
    data_set_input = boston['data']
    data_set_output = boston['target'][:, np.newaxis]
    # 1-2 数据归一化
    ss_input = MinMaxScaler()
    ss_output = MinMaxScaler()
    data_set_input = ss_input.fit_transform(boston['data'])
    data_set_output = ss_output.fit_transform(boston['target'][:, np.newaxis])
    # 1-3 打乱数据集
    index = [i for i in range(len(data_set_input))]
    random.seed(4)
    random.shuffle(index)
    data_set_input = data_set_input[index]
    data_set_output = data_set_output[index]

    # 1-4 划分数据集
    train_set_input = data_set_input[0:350, :]
    train_set_output = data_set_output[0:350, :]
    test_set_input = data_set_input[350:, :]
    test_set_output = data_set_output[350:, :]
    # 2 构建网络
    net = nn.Sequential(
    nn.Linear(13, 14),
    nn.Tanh(),
    nn.Linear(14, 1),
    nn.Sigmoid()
    )

    # 3 定义优化器和损失函数
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 4 训练网络
    loss_step = True
    l_arr = []
    for i in range(max_epoch):
        prediction = net(torch.FloatTensor(train_set_input))
        l = loss(torch.FloatTensor(train_set_output), prediction)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_arr.append(l.item())
        if i % 1 == 0:
            print('BP_step:{},loss:{}'.format(i, l_arr[i]))
            # logger.info('BP_step:{},loss:{}'.format(i,l_arr[i]))
        if loss_step and l_arr[i] <= 0.01:
            print('BP迭代从第 {} 步开始误差小于0.01'.format(i))
            # logger.info('BP迭代从第 {} 步开始误差小于0.01'.format(i))
            loss_step = False
    y_train_predict= prediction.detach().numpy().reshape(-1, 1)
    y_train_real = np.array(train_set_output).reshape(-1, 1)
    #5 测试网络效果
    predict = net(torch.FloatTensor(test_set_input))
    test_loss = loss(torch.FloatTensor(test_set_output), predict)
    print('train_loss:%.6f , test_loss:%.6f' % (torch.FloatTensor(l_arr).min(), test_loss))

    return test_set_output, predict, y_train_predict,y_train_real,l_arr


if __name__ == '__main__':
    data = load_boston()
    test_set_output, predict,y_train_predict,y_train_real, l_arr = BP(data)
    # predict = predict.detach().numpy().reshape(-1,1)
    # real = np.array(test_set_output).reshape(-1,1)
    # scaler = MinMaxScaler()
    # predict = scaler.fit(predict)
    # predict = scaler.inverse_transform(predict)
    # real = scaler.fit(real)
    # real = scaler.inverse_transform(real)
    # mse(real, predict)
    # rmse(real, predict)
    # mae(real, predict)
    # TV(real, predict)
    # TSD(real, predict)
