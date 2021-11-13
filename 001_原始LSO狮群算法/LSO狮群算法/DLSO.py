import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb
# 狮群算法的机制改进和应用研究
''' 种群初始化函数 '''


def initial(pop, dim, ub, lb):
    '''
    采用tent映射初始化
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



'''狮群算法'''


def DLSO(pop, dim, lb, ub, MaxIter, Maxstep, fun):
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
    XhisBest = copy.copy(X)
    fithisBest = copy.copy(fitness)
    indexBest = index
    gbest = copy.copy(GbestPositon)  # 全局最优
    Curve = np.zeros([MaxIter, 1])
    GbestScore_step = []
    GbestPositon_step = []
    for step in range(Maxstep):

        for t in range(MaxIter):

            # 母狮移动范围扰动因子计算
            stepf = 0.1 * (np.mean(ub) - np.mean(lb))
            alphaf = stepf * np.exp(-30 * t / MaxIter) ** 10
            # 幼狮移动范围扰动因子计算
            alpha = (MaxIter - t) / MaxIter
            # 差分扰动因子
            F0 = 0.8
            alphaF = np.exp(1 - (MaxIter / (MaxIter+1-t)))
            F = F0 * pow(2, alphaF)
            # 母狮位置更新
            for i in range(Nc):
                q = np.random.random()
                index = index1 = i
                while index == i:
                    index = np.random.randint(Nc)  # 随机挑选一只母狮
                while index1 == i:
                    index1 = index2 = np.random.randint(pop) # 随机挑选一只除狮王外的狮子
                while index2 == index1 or index1 == i:
                    index2 = np.random.randint(pop)

                if q <= 1/3:
                    X[i, :] = (XhisBest[i, :] + XhisBest[index, :]) * (1 + alphaf * np.random.randn(dim)) / 2
                elif q <= 2/3 and q > 1/3:
                    X[i, :] = XhisBest[i, :] + F * (gbest - XhisBest[i, :])
                else:
                    X[i, :] = XhisBest[i, :] + F * (XhisBest[index1, :] - XhisBest[index2, :])
            # 幼师位置更新
            for i in range(Nc, pop):
                q = np.random.random()
                nabla = 1 - (t * t) / (MaxIter * MaxIter)

                if q <= 0.5 * nabla:
                    X[i, :] = (gbest + XhisBest[i, :]) * (1 + alpha * np.random.randn(dim)) / 2
                elif q > 0.5 * nabla and q <= nabla:
                    indexm1 = np.random.randint(Nc)  # 随机挑选一只母狮
                    X[i, :] = (XhisBest[indexm1, :] + XhisBest[i, :]) * (1 + alpha * np.random.randn(dim)) / 2
                else:
                    idx = np.array(random.sample(range(pop),3)) # 随机取三个不同的狮子
                    X[i, :] = XhisBest[idx[0], :] + F * (XhisBest[idx[1], :] - XhisBest[idx[2], :])

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

            # 狮王更新
            Temp = np.zeros([1, dim])
            Temp[0, :] = gbest * (1 + np.random.randn(dim) * np.abs(XhisBest[indexBest, :] - gbest))
            Temp[0, :] = BorderCheck(Temp, ub, lb, 1, dim)  # 边界检测
            fitTemp = fun(Temp[0, :])  # 位置更新后的狮王当前适应度值
            # 更新后狮王的适应度值如果优于当前迭代中的全局最优适应度值，更新全局最优值，和全局最优位置
            if fitTemp < GbestScore:
                GbestScore = copy.copy(fitTemp)
                GbestPositon = copy.copy(Temp)
                fitness[indexBest] = copy.copy(fitTemp)
                X[indexBest, :] = copy.copy(Temp)
            # if GbestScore != 0:
            print('第{}次最佳适应值为{}'.format(t, GbestScore))
            value = np.min(fitness)  # 找最小值
            index = np.argmin(fitness)  # 最小值位置索引
            gbest = copy.copy(X[index, :])
            Curve[t] = GbestScore

        GbestScore_step.append(GbestScore)
        GbestPositon_step.append(GbestPositon)
        # GbestScore_mean = np.mean(GbestScore_step)
        # GbestPositon_mean = np.mean(GbestPositon_step)
        # GbestScore_max = np.max(GbestScore)
        # GbestPositon_max = np.max(GbestPositon)
    # print('DLSO运行30次的最优值：', np.array(GbestScore_step).reshape(1, -1))
    GbestScore = np.min(GbestScore_step)
    GbestPositon = GbestPositon_step[GbestScore_step.index(GbestScore)]

    return GbestScore, GbestPositon, Curve
    # return GbestScore,GbestPositon,GbestScore_max,GbestPositon_max,GbestScore_mean,GbestPositon_mean,Curve
