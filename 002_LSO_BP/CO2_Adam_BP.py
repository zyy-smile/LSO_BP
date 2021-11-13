import torch
from torch.utils.data import DataLoader,TensorDataset
from visdom import Visdom
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import xlrd
import numpy as np
import pdb
import time
from Dataset_preprocess import Data_Preprocess

batch_size = 4
learning_rate = 0.01
epochs = 50000
device = torch.device('cuda')
CO2_dataset = False
begin_time = time.time()

# 读取数据集
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

if CO2_dataset:
    pathX = r'D:\狮群算法\002_LSO_BP\CPS_85_Wages_data.xls'  # [534,11]
    bp_data = excel2matrix(pathX)
    # 划分训练集和测试集
    train_loader = bp_data[0:500,:]   #shape[25,6]
    test_loader = bp_data[500:534,:]   #shape[15,6]
    # 将numpy数据转化为tensor数据
    train_loader = torch.from_numpy(train_loader).float()
    test_loader = torch.from_numpy(test_loader).float()

    x_train = train_loader[:, 0:10]
    y_train = train_loader[:, 10].view(500,1)
    x_test = test_loader[:,0:10]
    y_test = test_loader[:,10].view(34,1)

else:
    path = u'加州房价数据集.csv'
    train_data, train_label, test_data, test_label = Data_Preprocess(path, )
    # 数据均一化

    # # 将numpy数据转化为tensor数据
    x_train = torch.from_numpy(train_data).float()
    y_train = torch.from_numpy(train_label).float().view(-1,1)
    x_test = torch.from_numpy(test_data).float()
    y_test = torch.from_numpy(test_label).float().view(-1,1)
# 搭建神经网络
input_num = x_train.shape[1]
hidden_num = 6
output_num = y_train.shape[1]
net1 = torch.nn.Sequential(
    torch.nn.Linear(input_num, hidden_num),
    torch.nn.Tanh(),    #激活函数
    torch.nn.Linear(hidden_num, output_num),
    )
# pdb.set_trace()
optimizer = optim.Adam(net1.parameters(),lr=learning_rate)  # 梯度下降优化器选择SGD
criterion = nn.MSELoss()  # 误差采用均方误差

print(net1)
print('optimier = Adam')
# 训练
losses = []
costs = []
# 训练10000轮
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

    if step % 1000 == 0:

        print('step{}:loss={}'.format(step,loss))
for i in range(len(losses)):
    cost = losses[i].item()
    costs.append(cost)
# print(costs)
end_time = time.time()
print(end_time-begin_time)
# save model
torch.save(net1.state_dict(),'net1.pt')
#  test
m_state_dict = torch.load('net1.pt')
# new_m = net1.to(device)
net1.load_state_dict(m_state_dict)
predict = net1(x_test)
print('predict=',predict.view(1,-1))
print('y_test=',y_test.view(1,-1))
test_Loss = criterion(y_test,predict)
print('test_loss=',test_Loss)
fig = plt.figure()
plt.plot(range(epochs),costs)
plt.title('Adam_loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()


