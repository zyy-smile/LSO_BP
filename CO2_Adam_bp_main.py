#-*- coding: UTF-8 -*- 
import torch
from torch.utils.data import DataLoader,TensorDataset
#from visdom import Visdom
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import xlrd
import numpy as np
import pdb
from torch.nn.parameter import Parameter
from logger import logger


batch_size = 4
learning_rate = 0.01
epochs = 100
device = torch.device('cuda')
pathX = r'BP1.xlsx'


def excel2matrix(path):
    '''
    将excel数据转化为numpy数据
    :param path: excel数据的存放位置
    :return: 转化后的numpy数组
    '''
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows, ncols))
    for i in range(nrows):
        rows = table.row_values(i)
        datamatrix[i,:] = rows
    return datamatrix

def data_preprocess(pathX):
    '''
    数据预处理
    :param pathX: 数据位置
    :return: 处理好的数据
    '''
    bp_data = excel2matrix(pathX)
    # 划分训练集和测试集
    train_loader = bp_data[0:25, 1:7]  # shape[25,6]
    test_loader = bp_data[25:40, 1:7]  # shape[15,6]

    # 将numpy数据转化为tensor数据
    train_loader = torch.from_numpy(train_loader)
    test_loader = torch.from_numpy(test_loader)
    train_loader = train_loader.float()
    test_loader = test_loader.float()

    x_train = train_loader[:, 0:5]
    y_train = train_loader[:, 5]
    y_train = y_train.view(25, 1)
    # y_train = y_train.squeeze(-1)
    x_test = test_loader[:, 0:5]
    # print(x_train.type(),y_train.type())
    y_test = test_loader[:, 5]
    y_test = y_test.view(15, 1)

    return x_train,y_train,x_test,y_test
# w:[w1, w2]  len(w[1]) = 4*5  len(w2) = 1*4 
# b:[b1, b2]  len(b1) = 4 len(b2) = 1
# ---------------------------------------

# y_test = y_test.squeeze(-1)
# print(y_test.type(),y_train.type())

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
    print(X)
    return X


def data_LSOToBP(X, input_num, hidden_num, output_num):
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


def BPNN(pathX, w, b, epochs=10000, learning_rate=learning_rate,need_paint=False,need_loss_log=False):
    '''
    神经网络训练
    :param pathX: 数据存放位置
    :param w: 权值矩阵
    :param b: 偏差矩阵
    :param epochs: 最大迭代次数
    :param need_paint: 是否需要打印
    :return: 测试集训练误差
    '''

    # 搭建神经网络
    x_train, y_train, x_test, y_test = data_preprocess(pathX)
    net1 = torch.nn.Sequential(torch.nn.Linear(5, 4),
                               torch.nn.Tanh(),  # 激活函数
                               torch.nn.Linear(4, 1),
                               torch.nn.Tanh())

    # 修改初始权值阈值
    net1[0].weight = Parameter(torch.Tensor(w[0]))
    net1[0].bias = Parameter(torch.Tensor(b[0]))
    net1[2].weight = Parameter(torch.Tensor(w[1]))
    net1[2].bias = Parameter(torch.Tensor(b[1]))
    
    optimizer = optim.Adam(net1.parameters(),lr=learning_rate)  # 梯度下降优化器选择Adam
    criterion = nn.MSELoss()  # 误差采用均方误差

    # 训练
    losses = []
    costs = []
    for step in range(epochs):
        prediction = net1(x_train)
        # print(prediction.type())
        loss = criterion(y_train,prediction)
        losses.append(loss)
        if torch.isnan(loss):
            break
        optimizer.zero_grad()  # 消除以前的梯度信息
        loss.backward()
        optimizer.step()
        # viz.line([float(loss)],[step],win='Adam_loss',opts=dict(title='adam_loss'),update='append')
        if step % 50 == 0:
            print('step{}:loss={}'.format(step,loss))
        if need_loss_log:
            logger.info("step{}:loss={}".format(step,loss))

    for i in range(len(losses)):
        cost = losses[i].item()
        costs.append(cost)

    # save model
    torch.save(net1.state_dict(),'net1.pt')

    #  test
    # m_state_dict = torch.load('net1.pt')
    # net1.load_state_dict(m_state_dict)
    # predict = net1(x_test)
    # test_Loss = criterion(y_test,predict)
    # print('test_loss=',test_Loss)
    
    if need_paint:
        fig = plt.figure()
        plt.plot(range(epochs),costs)
        plt.title('Adam_loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.show()
       
    #TODO: test_Loss convert to fitness
    return float(losses[-1])

def fitness_function(X):
    # 计算适应度函数
    net1 = torch.nn.Sequential(torch.nn.Linear(5, 4),
                               torch.nn.Tanh(),  # 激活函数
                               torch.nn.Linear(4, 1),
                               torch.nn.Tanh())
    return




if __name__ == '__main__':
    # 狮群参数设置

    w1 = np.random.random((4,5))
    w2 = np.random.random((1,4))
    w = [w1, w2]
    b1 = np.random.random((1,4))
    b2 = np.random.random((1,1))
    b = [b1, b2]
    X = data_BPToLSO(w,b)


    # test_loss = BPNN(pathX, w, b,epochs=10000,need_paint=True)
    # print(test_loss)
