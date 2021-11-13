import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
''' 种群初始化函数 '''


def initial(pop, dim, ub, lb):
    '''

    :param pop: 种群数量
    :param dim: 特征长度（测试函数的维度）
    :param ub: 取值范围的上限
    :param lb: 取值范围的下限
    :return:
    '''

    X = np.zeros([pop, dim])
    # 根据狮群的移动范围（取值范围）随机生成初始种群
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub


'''边界检查函数'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''


def CaculateFitness(X, fun):
    '''

    :param X: 物种信息，X=>[pop,dim]
    :param fun: 测试函数
    :return: 每个种群的适应度
    '''
    pop = X.shape[0]  # 种群数量
    # 适应度初始化，刚开始全部赋值为0
    fitness = np.zeros([pop, 1])
    # 计算每个种群的适应度
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''适应度排序'''


def SortFitness(Fit):
    '''

    :param Fit:还未排序的适应度
    :return: 排序好的适应度值及其索引
    '''
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

def dynamic_graph(populations, best_fitness, MaxIter):
    '''画图-进化动态以及进化曲线
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    ub = 5
    x1 = []
    x2 = []
    for i, j in populations:
        x1.append(i)
        x2.append(j)

    plt.ion()
    plt.figure(1)
    # 进化动态图
    plt.subplot(121)
    plt.plot(x1, x2, 'ro', markersize=2)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.title('The {} th optimization process of LSO'.format(len(best_fitness)))
    # plt.text(1,1, 'x=1')
    plt.axis([-ub, ub, -ub, ub])
    plt.grid(True)
    # 进化曲线
    plt.subplot(122)
    plt.plot(best_fitness, 'r')
    plt.xlabel('iterations times')
    plt.ylabel(' Optimal Value')
    plt.title('The optimal value varies with iteration')
    plt.axis([1, 100, 0, 1])
    plt.draw()
    plt.savefig("pictures/LSO_picture/{}_{}.png".format(len(best_fitness),str(int(time.time()))))
    plt.pause(0.1)
    plt.ioff()
    # plt.close()
    plt.clf()

'''狮群算法'''


def LSO(pop, dim, lb, ub, MaxIter, fun, gif = False):
    beta = 0.2  # 成年狮所占比列
    Nc = round(pop * beta)  # 成年狮数量
    Np = pop - Nc  # 幼师数量
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    value = np.min(fitness)  # 找最小值
    index = np.argmin(fitness)  # 最小值位置索引
    GbestScore = value  # 记录最好的适应度值
    # copy.copy浅复制:将原有的数据块打上一个新标签，所以当其中一个标签被改变的时候，数据块就会发生
    GbestPositon = copy.copy(X[index, :])  # 记录最好的位置，GbestPositon会随着X[index,:]的改变而改变
    XhisBest = copy.copy(X)  # 记录每个种群历史最优位置
    fithisBest = copy.copy(fitness)  # 记录每个种群历史最优值
    indexBest = index
    gbest = copy.copy(GbestPositon)
    Curve = []
    mushi_list = list(range(pop))
    mushi_list.remove(index)
    mushi_position = random.sample(mushi_list, Nc-1)     #母狮位置
    label = True
    for t in range(MaxIter):
        # 狮王更新
        Temp = np.zeros([1, dim])
        Temp[0, :] = gbest * (1 + np.random.randn(dim) * np.abs(XhisBest[indexBest, :] - gbest))
        Temp[0, :] = BorderCheck(Temp, ub, lb, 1, dim)  # 边界检测
        fitTemp = fun(Temp[0, :])  # 位置更新后的狮王当前适应度值
        fitness[indexBest] = copy.copy(fitTemp)
        X[indexBest, :] = copy.copy(Temp)    # 更新狮王位置
        # 更新后狮王的适应度值如果优于当前迭代中的全局最优适应度值，更新全局最优值，和全局最优位置
        if fitTemp < GbestScore:
            GbestScore = copy.copy(fitTemp)
            GbestPositon = copy.copy(Temp)


        # 母狮移动范围扰动因子计算
        stepf = 0.1 * (np.mean(ub) - np.mean(lb))
        alphaf = stepf * np.exp(-30 * t / MaxIter) ** 10
        # 幼狮移动范围扰动因子计算
        alpha = (MaxIter - t) / MaxIter
        # 母狮位置更新
        for i in mushi_position:
            index = i
            while index == i:
                index = random.sample(mushi_position, 1)[0] # 随机挑选一只母狮
            X[i, :] = (XhisBest[i, :] + XhisBest[index, :]) * (1 + alphaf * np.random.randn(dim)) / 2
        # 幼师位置更新
        for i in range(pop):
            if i == indexBest or i in mushi_position:
                continue
            q = np.random.random()
            if q < 1 / 3:
                X[i, :] = (gbest + XhisBest[i, :]) * (1 + alpha * np.random.randn(dim)) / 2
            elif q > 1 / 3 and q < 2 / 3:
                index1 = np.random.randint(1, Nc)
                X[i, :] = (XhisBest[index1, :] + XhisBest[i, :]) * (1 + alpha * np.random.randn(dim)) / 2
            else:
                gbestT = ub.T + lb.T - gbest;
                X[i, :] = (gbestT + XhisBest[i, :]) * (1 + alpha * np.random.randn(dim)) / 2

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        for j in range(pop):
            # 更新在k次迭代中该个体历史最优值
            if fitness[j] < fithisBest[j]:  # 更新个体历史最优值
                XhisBest[j, :] = copy.copy(X[j, :])
                fithisBest[j] = copy.copy(fitness[j])

            # 更新第k次迭代中的全局最优值
            if fitness[j] < GbestScore:  # 当前的适应度小于目前全局最优适应度值，更新全局最优适应度值
                GbestScore = copy.copy(fitness[j])
                GbestPositon = copy.copy(X[j, :])
                indexBest = j
        # if GbestScore <= 0:
        #     break
        # print('第{}次最优值{}'.format(t, GbestScore))
        if False :  # 提前结束条件
            # todo
            pass

        gbest = copy.copy(X[index, :])
        # 重新排序，确定位置
        if t % 10== 0 :
            indexBest = np.argmin(fitness)  # 狮王位置
            mushi_list = list(range(pop))
            mushi_list.remove(indexBest)
            mushi_position = random.sample(mushi_list, Nc - 1)  # 母狮位置


        if label and GbestScore == 0:
            print("LSO第{}次达到0".format(t))
            label = False
        Curve.append(GbestScore)
        if gif :
            dynamic_graph(X, Curve, MaxIter)

    return GbestScore, GbestPositon, Curve


