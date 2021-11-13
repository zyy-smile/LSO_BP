import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main_1 import fun1, fun2, fun15 , preprocess_X
import CLSO_2
import time

''' 种群初始化函数 '''
dim = 2
def fun8(X):  # Rastrigin 全局最小值为所有变量均为0时取得O=0 Xi=0 f(x)=0
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
    X = preprocess_X(X)
    O = 10 * dim + np.sum(X*X - 10*np.cos(2*np.pi*X))
    return O

def initial(pop, dim, ub, lb):
    '''

    :param pop: 种群数量
    :param dim: 特征长度（测试函数的维度）
    :param ub: 取值范围的上限
    :param lb: 取值范围的下限
    :return:
    '''

    X = np.zeros([pop, dim])
    for i in range(pop):
        X[i][0] = random.random()
    # 根据狮群的移动范围（取值范围）随机生成初始种群
    # for i in range(pop):
    #     for j in range(dim):
    #         X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
    for i in range(pop):
        for j in range(dim - 1):
            if X[i, j] == 0 or X[i, j] == 0.25 or X[i, j] == 0.5 or X[i, j] == 0.75:
                X[i, j] = X[i, j] + 0.1 * random.random()
            if 0 <= X[i, j] < 0.5:
                X[i, j + 1] = 2 * X[i, j]
            elif 0.5 < X[i, j] <= 1:
                X[i, j + 1] = 2 * X[i, j]
    for i in range(pop):
        for j in range(dim):
            X[i, j] = X[i, j] * (ub[j] - lb[j]) + lb[j]
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
    # import pdb
    # pdb.set_trace()
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


'''混沌搜索'''


def Chao_search(X, Fit, fun):
    # 对当前所有粒子按适应度排序
    pop = X.shape[0]
    dim = X.shape[1]
    fitness, index = SortFitness(Fit)
    X_new = SortPosition(X, index)
    # 取80%的粒子
    row = int(X_new.shape[0] * 0.8)
    X_cnew = np.array(X_new[:row, :])  # [0.8pop, dim]
    # 求该粒子的每列最小值a和最大值b
    a = b = []  # a,b => [1,dim]
    a = np.min(X_cnew, axis=0)
    b = np.max(X_cnew, axis=0)
    a.reshape(1, -1)
    b.reshape(1, -1)
    # 最优位置pg映射到(0,1)区间pg => z
    z = [0 for i in range(dim)]
    pg = X_new[0]  # pg => [1,dim]
    for i in range(len(pg)):
        z[i] = (pg[i] - a[i]) / (b[i] - a[i])
    # 根据z[i]生成混沌映射 z => zm
    zm = [[0 for i in range(dim)] for i in range(row)]  # zm => [pop*0.8, dim]
    for i in range(len(z)):
        zm[0][i] = z[i]
    for i in range(X_cnew.shape[1]):
        for j in range(X_cnew.shape[0] - 1):
            if 0 <= zm[j][i] <= 0.5:
                zm[j + 1][i] = 2 * zm[j][i]
            elif 0.5 < zm[j][i] <= 1:
                zm[j + 1][i] = 2 * zm[j][i]

    zm = np.array(zm)
    # 映射到种群范围 zm => pgm  [0.8pop, dim]
    pgm = [[0 for i in range(dim)] for i in range(row)]
    for i in range(zm.shape[1]):
        for j in range(zm.shape[0]):
            pgm[j][i] = (zm[j][i] - a[i]) / (b[i] - a[i])
    # 将原种群与新生成的混沌种群混合，选择较优的pop个种群构成新种群
    X_cnew = np.array(pgm)
    X_sum = np.vstack((X_new, X_cnew))  # [pop+0.8pop, dim]
    fitness = CaculateFitness(X_sum, fun)
    fitness, index = SortFitness(fitness)
    X_cnew = SortPosition(X_sum, index)
    X_cnew = X_cnew[:pop, :]
    fitness = fitness[:pop, :]

    return X_cnew, fitness

# 动态二维图
def dynamic_graph(populations, best_fitness, need_save=False):
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
    plt.title('GLSO第{}次寻优过程'.format(len(best_fitness)))
    # plt.text(1,1, 'x=1')
    plt.axis([-ub, ub, -ub, ub])
    plt.grid(True)
    # 进化曲线
    plt.subplot(122)
    plt.plot(best_fitness, 'r')
    plt.xlabel('迭代次数')
    plt.ylabel('最优值')
    plt.title('最优值随迭代变化')
    plt.axis([1, 100, 0, 1])
    # plt.close()
    plt.draw()
    if need_save:
        plt.savefig("pictures/Chao_picture/{}.png".format(len(best_fitness)))
    plt.pause(0.1)
    plt.ioff()
    plt.clf()



'''狮群算法'''


def CLSO(pop, dim, lb, ub, MaxIter, fun):
    populations =[]
    beta = 0.2  # 成年狮所占比列
    Nc = round(pop * beta)  # 成年狮数量
    Np = pop - Nc  # 幼师数量
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    X = BorderCheck(X, ub, lb, pop, dim)
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
    mushi_position = random.sample(mushi_list, Nc - 1)  # 母狮位置

    for t in range(MaxIter):
        # 狮王更新
        Temp = np.zeros([1, dim])
        Temp[0, :] = gbest * (1 + np.random.randn(dim) * np.abs(XhisBest[indexBest, :] - gbest))
        Temp[0, :] = BorderCheck(Temp, ub, lb, 1, dim)  # 边界检测
        fitTemp = fun(Temp[0, :])  # 位置更新后的狮王当前适应度值
        fitness[indexBest] = copy.copy(fitTemp)
        X[indexBest, :] = copy.copy(Temp)  # 更新狮王位置
        # 更新后狮王的适应度值如果优于当前迭代中的全局最优适应度值，更新全局最优值，和全局最优位置
        if fitTemp < GbestScore:
            GbestScore = copy.copy(fitTemp)
            GbestPositon = copy.copy(Temp)

        # 母狮移动范围扰动因子计算
        stepf = 0.1 * (np.mean(ub) - np.mean(lb))
        alphaf = stepf * np.exp(-30 * t / MaxIter) ** 10
        # 幼狮移动范围扰动因子计算
        alpha = (MaxIter - t) / MaxIter
        # 差分扰动因子
        F0 = 0.8
        alphaF = np.exp(1 - (MaxIter / (MaxIter + 1 - t)))
        F = F0 * pow(2, alphaF)
        # 采用灰狼更新的母狮参数
        a = 2 * (MaxIter - t) / MaxIter
        # 母狮位置更新
        # 前三个最优母狮位置索引
        mushi_idx = fitness.argsort()[:3]
        for i in mushi_position:
            index = i  # 当前需要更新位置的母狮索引
            q = np.random.random()
            while index == i:
                index = random.sample(mushi_position, 1)[0]  # 随机挑选一只母狮
            if q <= 1 / 3:
                X[i, :] = (XhisBest[i, :] + XhisBest[index, :]) * (1 + alphaf * np.random.randn(dim)) / 2
            elif q <= 2 / 3 and q > 1 / 3:
                X_sum = []
                for j in mushi_idx:
                    A = 2 * a * np.random.random() - a
                    C = 2 * np.random.random()
                    x = XhisBest[j, :] - A * np.abs(C * XhisBest[j, :] - XhisBest[i, :])
                    X_sum.append(x)
                X[i, :] = np.mean(X_sum, axis=0)
                # X[i, :] = XhisBest[i, :] + F * (gbest - XhisBest[i, :])
            else:
                indx1 = random.sample(mushi_position, 1)
                while indx1 == i:
                    indx1 = random.sample(mushi_position, 1)
                indx2 = random.sample(mushi_position, 1)
                while indx2 == i or indx2 == indx1:
                    indx2 = random.sample(mushi_position, 1)
                X[i:] = XhisBest[i, :] + F * (XhisBest[indx1, :] - XhisBest[indx2, :])
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
                gbestT = ub.T + lb.T - gbest
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
        if False:  # 提前结束条件
            # todo
            pass

        gbest = copy.copy(X[index, :])

        # rerank = 0.2 * MaxIter - t
        # if rerank < 1:
        #     rerank = 1
        if t % 10 == 0:
            indexBest = np.argmin(fitness)  # 狮王位置
            mushi_list = list(range(pop))
            mushi_list.remove(indexBest)
            mushi_position = random.sample(mushi_list, Nc - 1)  # 母狮位置

        Curve.append(GbestScore)
        dynamic_graph(X, Curve)

    return GbestScore, GbestPositon, Curve


if __name__ == '__main__':
    pop = 100
    dim = 2
    MaxIter = 100
    fobj = fun8
    lb = -5
    ub = 5
    lb = lb * np.ones([dim, 1])  # 下边界
    ub = ub * np.ones([dim, 1])  # 上边界
    GLSO_GbestScore, GLSO_GbestPositon, GLSO_Curve = CLSO(pop, dim, lb, ub, MaxIter, fobj)
    print('CLSO最优适应度值：', GLSO_GbestScore)