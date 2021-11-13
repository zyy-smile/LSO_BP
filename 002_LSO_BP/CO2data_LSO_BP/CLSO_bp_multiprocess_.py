import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CO2_Adam_bp_main import BPNN, data_LSOToBP, pathX, data_BPToLSO
from logger import logger
import pdb
from multiprocessing import Pool
import os,time,random

''' 种群初始化函数 '''

input_num = 5
hidden_num = 4
output_num = 1
pool = None

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
def cal_fitness(x, fun, order=0, fitness=None):
    w, b = data_LSOToBP(x, input_num, hidden_num, output_num)
    res = fun(pathX, w, b, epochs=100, learning_rate=0.02)
    if fitness is not None:
        fitness[order]=res
    return res


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
        fitness[i] = cal_fitness(X[i, :], fun)
    return fitness



def CaculateFitness_parallel(X, fun):
    pool = Pool(7)
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for idx, x in enumerate(X):
        pool.apply_async(cal_fitness,(x, fun, idx, fitness))
    pool.close() #关闭进程池，关闭后po不再接收新的请求
    pool.join()

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
def Chao_search(X,Fit,fun):
    # 对当前所有粒子按适应度排序
    pop = X.shape[0]
    dim = X.shape[1]
    fitness, index = SortFitness(Fit)
    X_new = SortPosition(X, index)
    # 取80%的粒子
    row = int(X_new.shape[0] * 0.8)
    X_cnew = np.array(X_new[:row,:])   # [0.8pop, dim]
    # 求该粒子的每列最小值a和最大值b
    a = b = []  # a,b => [1,dim]
    a = np.min(X_cnew,axis=0)
    b = np.max(X_cnew,axis=0)
    a.reshape(1,-1)
    b.reshape(1,-1)
    # 最优位置pg映射到(0,1)区间pg => z
    z = [0 for i in range(dim)]
    pg = X_new[0]  #pg => [1,dim]
    for i in range(len(pg)):
        z[i] = (pg[i] - a[i]) / (b[i] - a[i])
    # 根据z[i]生成混沌映射 z => zm
    zm = [[0 for i in range(dim)] for i in range(row)]  # zm => [pop*0.8, dim]
    for i in range(len(z)):
        zm[0][i] = z[i]
    for i in range(X_cnew.shape[1]):
        for j in range(X_cnew.shape[0]-1):
            if 0<= zm[j][i] <= 0.5:
                zm[j+1][i] = 2 * zm[j][i]
            elif 0.5 < zm[j][i] <= 1:
                zm[j+1][i] = 2 * zm[j][i]

    zm = np.array(zm)
    # 映射到种群范围 zm => pgm  [0.8pop, dim]
    pgm = [[0 for i in range(dim)] for i in range(row)]
    for i in range(zm.shape[1]):
        for j in range(zm.shape[0]):
            pgm[j][i] = (zm[j][i] - a[i]) / (b[i] - a[i])
    # 将原种群与新生成的混沌种群混合，选择较优的pop个种群构成新种群
    X_cnew = np.array(pgm)
    X_sum = np.vstack((X_new, X_cnew))  # [pop+0.8pop, dim]
    fitness = CaculateFitness_parallel(X_sum, fun)
    fitness, index = SortFitness(fitness)
    X_cnew = SortPosition(X_sum, index)
    X_cnew = X_cnew[:pop,:]
    fitness = fitness[:pop,:]

    return X_cnew,fitness

'''狮群算法'''


def CLSO(pop, dim, lb, ub, MaxIter, Maxstep, fun):
    beta = 0.2  # 成年狮所占比列
    Nc = round(pop * beta)  # 成年狮数量
    Np = pop - Nc  # 幼师数量
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    X = BorderCheck(X, ub, lb, pop, dim)
    fitness = CaculateFitness_parallel(X, fun)  # 计算适应度值
    value = np.min(fitness)  # 找最小值
    index = np.argmin(fitness)  # 最小值位置索引
    GbestScore = value  # 记录最好的适应度值
    # copy.copy浅复制:将原有的数据块打上一个新标签，所以当其中一个标签被改变的时候，数据块就会发生
    GbestPositon = copy.copy(X[index, :])  # 记录最好的位置，GbestPositon会随着X[index,:]的改变而改变
    XhisBest = copy.copy(X)  # 记录每个种群历史最优位置
    fithisBest = copy.copy(fitness)  # 记录每个种群历史最优值
    indexBest = index
    gbest = copy.copy(GbestPositon)
    Curve = np.zeros([MaxIter, 1])
    mushi_list = list(range(pop))
    mushi_list.remove(index)
    mushi_position = random.sample(mushi_list, Nc-1)     #母狮位置

    for t in range(MaxIter):
        # 狮王更新
        logger.info('CLSO第{}次迭代'.format(t))
        Temp = np.zeros([1, dim])
        Temp[0, :] = gbest * (1 + np.random.randn(dim) * np.abs(XhisBest[indexBest, :] - gbest))
        Temp[0, :] = BorderCheck(Temp, ub, lb, 1, dim)  # 边界检测
        fitTemp = cal_fitness(Temp[0, :], fun) # 位置更新后的狮王当前适应度值
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
        # 差分扰动因子
        F0 = 0.8
        alphaF = np.exp(1 - (MaxIter / (MaxIter + 1 - t)))
        F = F0 * pow(2, alphaF)

        # 母狮位置更新
        for i in mushi_position:
            index = i
            q = np.random.random()
            while index == i:
                index = random.sample(mushi_position, 1)[0] # 随机挑选一只母狮
            if q <= 1 / 3:
                X[i, :] = (XhisBest[i, :] + XhisBest[index, :]) * (1 + alphaf * np.random.randn(dim)) / 2
            elif q <= 2 / 3 and q > 1 / 3:
                X[i, :] = XhisBest[i, :] + F * (gbest - XhisBest[i, :])
            else:
                indx1 = random.sample(mushi_position,1)
                while indx1 == i:
                    indx1 = random.sample(mushi_position, 1)
                indx2 = random.sample(mushi_position,1)
                while indx2 == i or indx2 == indx1:
                    indx2 = random.sample(mushi_position, 1)
                X[i :] = XhisBest[i, :] + F * (XhisBest[indx1, :] - XhisBest[indx2, :])
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
        fitness = CaculateFitness_parallel(X, fun)  # 计算适应度值

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

        logger.info('第{}次最优值{}'.format(t, GbestScore))
        if False :  # 提前结束条件
            # todo
            pass

        gbest = copy.copy(X[index, :])


        if t % 10 == 0 :
            indexBest = np.argmin(fitness)  # 狮王位置
            mushi_list = list(range(pop))
            mushi_list.remove(indexBest)
            mushi_position = random.sample(mushi_list, Nc - 1)  # 母狮位置



        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve

def train_with_init_para(parameter_str):
    parameter_str = parameter_str.replace('- ','-').replace('\n','')
    parameters_list = (parameter_str.strip())[1:-1].split(' ') # 去首尾两边空格及括号
    parameters_list = [float(el) for el in parameters_list if el]
    w, b = data_LSOToBP(parameters_list, input_num, hidden_num, output_num)

    MSE_loss = BPNN(pathX, w, b, epochs=10000, need_paint=True)
    print(MSE_loss)
    logger.info(MSE_loss)



if __name__ == '__main__':
    #pool = Pool(7)

 #    parameter_str = """[-0.52411906  1.41107021  0.27562307 -0.2388811  -1.17569658  0.06560239
 # -0.35542538 -0.15471994 -0.85810313 -2.44590858 -0.21169572 -0.39662391
 #  0.05728673  0.02480188  0.07678745 -0.62464596  0.58233787 -0.1510885
 #  0.36907253 -0.32006022 -0.23589334 -0.07722064 -0.34247963 -0.78432908
 # -0.1080313   1.38738103 -0.65063434  0.53649746  0.13509974]"""
 #    train_with_init_para(parameter_str)
 #    raise ValueError('')

    # import time
    #
    # parameter_str = """[0.18542244  0.03198981 - 0.01278663  0.2603193   1.05257651 - 1.20712673
    #  - 0.03752174 - 0.36287946 - 0.34913694 - 0.24395438 - 0.13709459 - 1.17395838
    #  - 0.16033736 - 1.24973085 - 0.06103344  0.6553006 - 0.47742225 - 0.00924062
    #  0.34801047 - 1.82865235  0.17509165 - 0.13101313 - 0.27743837 - 0.01959079
    #  - 0.17992199  0.50692621  0.05033401 - 1.32495183 - 0.05525384]"""
    # parameter_str = parameter_str.replace('- ', '-').replace('\n', '')
    # parameters_list = (parameter_str.strip())[1:-1].split(' ')  # 去首尾两边空格及括号
    # parameters_list = [float(el) for el in parameters_list if el]
    # begin_time = time.time()
    # w, b = data_LSOToBP(parameters_list, input_num, hidden_num, output_num)
    # res = BPNN(pathX, w, b, epochs=100, learning_rate=0.02)
    # end_time = time.time()
    # print("BPNN_Runtime2:",end_time-begin_time)
    # raise ValueError('')


    # 5*4*1 = 29
    pop = 50
    dim = 29
    lb = np.array([-5]*dim)
    ub = np.array([5]*dim)
    MaxIter = 100  #狮群迭代次数
    Maxstep = 1
    fun = BPNN
    import pdb
    pdb.set_trace()
    logger.info("狮群数量：{}狮群迭代次数：{}".format(pop,MaxIter))
    _, GbestPositon, _ = CLSO(pop, dim, lb, ub, MaxIter, Maxstep, fun)
    print(GbestPositon)
    logger.info(GbestPositon)
    w,b = data_LSOToBP(GbestPositon, input_num, hidden_num, output_num)
    MSE_loss = BPNN(pathX, w, b, epochs=1000, learning_rate=0.005, need_paint=True)
    print(MSE_loss)
    logger.info(MSE_loss)