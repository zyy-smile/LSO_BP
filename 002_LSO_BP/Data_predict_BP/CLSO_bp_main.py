import torch
import random
from torch.utils.data import DataLoader,TensorDataset
from visdom import Visdom
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import pdb
from torch.nn.parameter import Parameter
from logger import logger
from matplotlib.font_manager import FontProperties
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler

data = load_boston()
learning_rate =0.01
def data_preprocess(data):
    # 1-1 加载数据
    boston = data
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
    train_set_input = data_set_input[0:406, :]
    train_set_output = data_set_output[0:406, :]
    test_set_input = data_set_input[406:506, :]
    test_set_output = data_set_output[406:506, :]

    return train_set_input, train_set_output, test_set_input, test_set_output

def data_BPToLSO(w,b):
    '''
    :param w:
    :param b:
    :return:
    '''
    l = [w,b]
    def get_items(l, res):
        if type(l) not in [list, np.ndarray]:
            res.append(l)
            return
        for el in l:
            if type(l) in [list, np.ndarray]:
                get_items(el, res)
            else:
                res.append(el)
    X = []
    get_items(l, X)
    
    return X


def data_LSOToBP(X, input_num, hidden_num, output_num):
    '''
    :param X:
    :param input_num:
    :param hidden_num:
    :param output_num:
    :return:
    '''
    begin, end = 0, input_num*hidden_num
    w1 = np.array(X[begin:end]).reshape(hidden_num,input_num)
    begin = end
    end += hidden_num*output_num
    w2 = np.array(X[begin:end]).reshape(output_num,hidden_num)
    begin = end
    end += hidden_num
    b1 = np.array(X[begin:end]).reshape(1,hidden_num)
    begin = end
    end += output_num
    b2 = np.array(X[begin:end]).reshape(1,output_num)
    w = [w1,w2]
    b = [b1,b2]

    return w, b
def mse(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    # print("Errors: ", error)
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值

    # logger.info("Square Error : {}".format(squaredError))
    # logger.info("Absolute Value of Error : {} ".format(absError))
    Mse = sum(squaredError) / len(squaredError)
    logger.info("MSE = {}".format(Mse[0]))  # 均方误差MSE


def rmse(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    # print("Errors: ", error)

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    Rmse = np.sqrt(sum(squaredError) / len(squaredError))
    logger.info("RMSE = {}".format(Rmse[0]))  # 均方根误差RMSE


def mae(prediction, target):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    Mae = sum(absError) / len(absError)
    logger.info("MAE = {}".format(Mae[0])) # 平均绝对误差MAE


def TV(prediction, target):
    targetDeviation = []
    targetMean = sum(target) / len(target)  # target平均值
    for val in target:
        targetDeviation.append((val - targetMean) * (val - targetMean))
    Variance = sum(targetDeviation) / len(targetDeviation)
    logger.info("Target Variance = {}".format(Variance[0]))  # 方差


def TSD(prediction, target):
    targetDeviation = []
    targetMean = sum(target) / len(target)  # target平均值
    for val in target:
        targetDeviation.append((val - targetMean) * (val - targetMean))
    Standard_Deviation = np.sqrt(sum(targetDeviation) / len(targetDeviation))
    logger.info("Target Standard Deviation = {}".format(Standard_Deviation[0]))  # 标准差


def BPNN(data, w, b, epochs=10000, learning_rate=learning_rate, need_paint=False, need_loss_log=False, optimization_algorithm='GLSO_BP'):
    '''
    神经网络训练
    :param data: 数据
    :param w: 权值矩阵
    :param b: 偏差矩阵
    :param epochs: 最大迭代次数
    :param need_paint: 是否需要打印
    :return: 测试集训练误差
    '''

    # 数据处理
    loss_step = True
    x_train, y_train, x_test, y_test = data_preprocess(data)
    in_num = x_train.shape[1]
    hid_num = 14
    out_num = y_train.shape[1]
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    # 搭建神经网络
    net1 = torch.nn.Sequential(torch.nn.Linear(in_num, hid_num),
                               torch.nn.Tanh(),  # 激活函数
                               torch.nn.Linear(hid_num, out_num),
                               torch.nn.Sigmoid())
    # 修改初始权值阈值
    net1[0].weight = Parameter(torch.Tensor(w[0]))
    net1[0].bias = Parameter(torch.Tensor(b[0]))
    net1[2].weight = Parameter(torch.Tensor(w[1]))
    net1[2].bias = Parameter(torch.Tensor(b[1]))
    
    optimizer = optim.Adam(net1.parameters(),lr=learning_rate)  # 梯度下降优化器选择Adam
    criterion = nn.MSELoss()  # 误差采用均方误差
    # criterion = nn.loss
    # 训练
    losses = []
    costs = []
    for step in range(epochs):
        prediction = net1(x_train)
        loss = criterion(y_train,prediction)
        losses.append(loss)
        if torch.isnan(loss):
            break
        optimizer.zero_grad()  # 消除以前的梯度信息
        loss.backward()
        optimizer.step()
        if need_loss_log:
            # if step % 100 == 0:
            #     print('step{}:loss={}'.format(step, loss))
            logger.info("{}_step{}:loss={}".format(optimization_algorithm,step,loss))
            if loss_step and loss <= 0.01:
                logger.info('{}迭代从第 {} 步开始误差小于0.01'.format(optimization_algorithm, step))
                loss_step = False

    for i in range(len(losses)):
        cost = losses[i].item()
        costs.append(cost)

    # save model
    torch.save(net1.state_dict(),'net1.pt')

    

       
    if need_paint:
        # plt.figure(figsize=(14, 14),dpi=100)
        # train
        train_predict = prediction.detach().numpy().reshape(-1, 1)
        y_train = y_train.detach().numpy().reshape(-1, 1)
        x_t = np.arange(train_predict.shape[0]).reshape(-1,1)

        # 记录训练集的真实值与预测值
        # logger.info('{}_train_data = {}'.format(optimization_algorithm, y_train.reshape(1, -1)))
        # logger.info('{}_train_predict_data = {}'.format(optimization_algorithm, train_predict.reshape(1, -1)))
        # test
        m_state_dict = torch.load('net1.pt')
        net1.load_state_dict(m_state_dict)
        predict = net1(x_test)
        test_Loss = criterion(y_test,predict)
        # print('test_loss=',test_Loss.item())
        for i in range(len(predict)):
            predict[i] = predict[i].item()
        test_predict = predict.detach().numpy().reshape(-1,1)
        y_test = y_test.detach().numpy().reshape(-1,1)
        x = np.arange(test_predict.shape[0]).reshape(-1,1)

        # 记录测试集的真实值与预测值
        # logger.info('{}_test_data = {}'.format(optimization_algorithm, y_test.reshape(1,-1)))
        # logger.info('{}_test_predict_data = {}'.format(optimization_algorithm, test_predict.reshape(1,-1)))
        # 误差
        logger.info('{}_train_Loss : '.format(optimization_algorithm))
        mse(y_train, train_predict)
        rmse(y_train, train_predict)
        mae(y_train, train_predict)
        TV(y_train, train_predict)
        TSD(y_train, train_predict)
        logger.info('{}_test_Loss : '.format(optimization_algorithm))
        mse(y_test, test_predict)
        rmse(y_test, test_predict)
        mae(y_test, test_predict)
        TV(y_test, test_predict)
        TSD(y_test, test_predict)

        # 画图
        # 1、损失图
        plt.subplot(2, 2, 1)
        # plt.plot(range(epochs),costs,label='CLSO_BP')
        # myfont = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
        plt.plot(range(epochs), costs, label=optimization_algorithm)
        plt.title('The loss value changes with the number of iterations')
        plt.xlabel('iterations times')
        plt.ylabel(' iteration loses ')
        plt.yscale('log')

        # 2 训练集数据真实值与预测值
        if optimization_algorithm =='GLSO_BP':
            plt.subplot(2, 2, 2)
            myfont = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
            plt.scatter(y_train, train_predict, marker='o', c='', s=10, edgecolors='k')
            plt.plot(y_train, y_train, c='red')
            plt.title('The actual value of the training set is compared with the predicted value')
            plt.xlabel('True value of training set', fontproperties=myfont)
            plt.ylabel('Predicted value of training set', fontproperties=myfont)
            plt.legend()

            # 3.1 测试数据真实值与预测值
            plt.subplot(2, 2, 3)
            myfont = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
            plt.plot(x, test_predict, c='red',label='predict_data')
            plt.plot(x, y_test, c='yellow', label='real_data')
            plt.title('The actual value of testing set is compared with the predicted value')
            plt.xlabel('sample data')
            plt.ylabel('housing price')
            plt.legend()

            # 3.2 测试数据真实值与预测值
            plt.subplot(2, 2, 4)
            myfont = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
            plt.scatter(y_test, test_predict, marker='o',c='',s=10, edgecolors='k')
            plt.plot(y_test, y_test, c='red')
            plt.title('The actual value of the test set is compared with the predicted value')
            plt.xlabel('True value of testing set', fontproperties=myfont)
            plt.ylabel('Predicted value of testing set', fontproperties=myfont)
            plt.legend()
    #TODO: test_Loss convert to fitness
    return float(losses[-1])




# w1 = np.random.random((14,13))
# w2 = np.random.random((1,14))
# w = [w1, w2]
# b1 = np.random.random((1,14))
# b2 = np.random.random((1,1))
# b = [b1, b2]
# data = load_boston()

# BPNN(data, w, b, need_paint=True, epochs=1000)