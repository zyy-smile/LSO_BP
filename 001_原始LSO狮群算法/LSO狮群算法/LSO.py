import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
    
    return X,lb,ub
            
'''边界检查函数'''
def BorderCheck(X,ub,lb,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X
    
    
'''计算适应度函数'''
def CaculateFitness(X,fun):
    '''

    :param X: 物种信息，X=>[pop,dim]
    :param fun: 测试函数
    :return: 每个种群的适应度
    '''
    pop = X.shape[0]   # 种群数量
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
    return fitness,index


'''根据适应度对位置进行排序'''
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew


'''狮群算法'''
def LSO(pop,dim,lb,ub,MaxIter,Maxstep,fun):
    beta = 0.2           # 成年狮所占比列
    Nc = round(pop*beta) # 成年狮数量
    Np = pop-Nc          # 幼师数量
    X,lb,ub = initial(pop, dim, ub, lb) # 初始化种群
    fitness = CaculateFitness(X,fun) # 计算适应度值
    value  = np.min(fitness) # 找最小值
    index = np.argmin(fitness) # 最小值位置索引
    GbestScore = value        # 记录最好的适应度值
    # copy.copy浅复制:将原有的数据块打上一个新标签，所以当其中一个标签被改变的时候，数据块就会发生
    GbestPositon = copy.copy(X[index,:])  # 记录最好的位置，GbestPositon会随着X[index,:]的改变而改变
    XhisBest = copy.copy(X)          # 记录每个种群历史最优位置
    fithisBest = copy.copy(fitness) # 记录每个种群历史最优值
    indexBest = index
    gbest = copy.copy(GbestPositon)
    Curve = np.zeros([MaxIter,1])
    GbestScore_step = []
    GbestPositon_step = []
    for step in range(Maxstep):
        for t in range(MaxIter):

            #母狮移动范围扰动因子计算
            stepf = 0.1*(np.mean(ub) - np.mean(lb))
            alphaf = stepf*np.exp(-30*t/MaxIter)**10
            #幼狮移动范围扰动因子计算
            alpha = (MaxIter - t)/MaxIter
            #母狮位置更新
            for i in range(Nc):
                index = i
                while index == i:
                    index = np.random.randint(Nc)#随机挑选一只母狮
                X[i,:] = (X[i,:] + X[index,:])*(1 + alphaf*np.random.randn(dim))/2
            #幼师位置更新
            for i in range(Nc, pop):
                q = np.random.random()
                if q < 1 / 3:
                    X[i, :] = (gbest + XhisBest[i, :]) * (1 + alpha * np.random.randn(dim)) / 2
                elif q > 1 / 3 and q < 2 / 3:
                    X[i, :] = (X[i, :] + XhisBest[i, :]) * (1 + alpha * np.random.randn(dim)) / 2
                else:
                    gbestT = ub.T + lb.T - gbest;
                    X[i,:] = (gbestT + XhisBest[i,:])*( 1 + alpha*np.random.randn(dim))/2

            X = BorderCheck(X,ub,lb,pop,dim) #边界检测
            fitness = CaculateFitness(X,fun) #计算适应度值
            for j in range(pop):
                # 更新在k次迭代中该个体历史最优值
                if fitness[j]<fithisBest[j]: # 更新个体历史最优值
                    XhisBest[j,:] = copy.copy(X[j,:])
                    fithisBest[j] = copy.copy(fitness[j])

                # 更新第k次迭代中的全局最优值
                if fitness[j]<GbestScore:  # 当前的适应度小于目前全局最优适应度值，更新全局最优适应度值
                    GbestScore = copy.copy(fitness[j])
                    GbestPositon = copy.copy(X[j,:])
                    indexBest = j

            #狮王更新
            Temp = np.zeros([1,dim])
            Temp[0,:] = gbest*(1 + np.random.randn(dim)*np.abs(XhisBest[indexBest,:] - gbest))
            Temp[0,:] = BorderCheck(Temp,ub,lb,1,dim) #边界检测
            fitTemp = fun(Temp[0,:])  # 位置更新后的狮王当前适应度值
            # 更新后狮王的适应度值如果优于当前迭代中的全局最优适应度值，更新全局最优值，和全局最优位置
            if fitTemp < GbestScore:
                GbestScore = copy.copy(fitTemp)
                GbestPositon =  copy.copy(Temp)
                fitness[indexBest] = copy.copy(fitTemp)
                X[indexBest,:] = copy.copy(Temp)
            # if GbestScore != 0:
            print('第{}次最佳适应值为{}'.format(t, GbestScore))
            value  = np.min(fitness) #找最小值
            index = np.argmin(fitness) #最小值位置索引
            gbest = copy.copy(X[index,:])
            Curve[t] = GbestScore

        GbestScore_step.append(float(GbestScore))
        GbestPositon_step.append(GbestPositon)
    print('LSO运行30次的最优值：',np.array(GbestScore_step).reshape(1,-1))
    GbestScore = np.min(GbestScore_step)
    GbestPositon = GbestPositon_step[GbestScore_step.index(GbestScore)]

    
    return GbestScore,GbestPositon,Curve

if __name__ == '__main__':
    def fun1(X):  # Sphere Function Xi=0 f(x)=0  单峰函数，可用来考验算法的收敛速度
        O = np.sum(X * X + X * X)
        return O


    def fun2(X):  # SchwefelProblem  np.prod()函数用来计算所有元素的乘积, Xi=0 f(x)=0
        # 高峰单维函数，考验算法收敛速度和收敛精度
        O = np.sum(np.abs(X)) + np.prod(np.abs(X))
        # print(X.shape,O.shape)
        return O


    def fun3(X):
        O = 0
        for i in range(len(X)):
            O = O + np.square(np.sum(X[0:i + 1]))
        return O


    def fun4(X):
        O = np.max(np.abs(X))
        return O


    def fun5(X):  # Generalized Rosenbrock  Xi=1, f(x)=0
        # 该函数全局最优点位于一个平滑、狭长的抛物线形山谷内，由于函数为优化算法提供的信息比较有限，使算法很难辨别搜索方向，查找最优解也变得十分困难
        X_len = len(X)
        O = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1))
        return O


    def fun6(X):  # Step Xi=0 f(x)=0  单峰函数，可用来考验算法的收敛速度
        O = np.sum(np.square(np.abs(X + 0.5)))
        return O


    def fun7(X):  # Generalized Schwefel‘s  30维时理论最小值f(420.9687) = -12569.5  |Xi|<=500，2维时,理论最小值f( 420.96875, 420.96875 ) = -837.965759277
        # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
        O = -np.sum(X * np.sin(np.sqrt(np.abs(X))))
        return O


    def fun8(X):  # Rastrigin 全局最小值为所有变量均为0时取得O=0 Xi=0 f(x)=0
        # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
        O = 10 * dim + np.sum(X * X - 10 * np.cos(2 * np.pi * X))
        return O


    def fun9(X):  # Generalized Griewank function 全局最小f(x)=0,x=(0,...,0)
        # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
        X_len = len(X)
        O1 = 1
        for i in range(1, X_len):
            O1 = O1 * np.cos(X[i] / np.sqrt(i))
        O = 1 / 4000 * np.sum(X * X) - O1 + 1
        return O


    # “基于改进狮群算法和BP神经网络模型的房价预测”中的测试函数
    def fun10(X):  # weierstrass
        k_max = 20
        a = 0.5
        b = 3
        O1 = 0
        O2 = 0
        X_len = len(X)
        for i in range(X_len):
            for j in range(k_max + 1):
                O1 = O1 + np.power(a, j) * np.cos((2 * np.pi * np.power(b, j) * (X[i] + 0.5)))
        for k in range(k_max + 1):
            O2 = O2 + np.power(a, k) * np.cos(np.pi * np.power(b, k))

        O = O1 - dim * O2

        return O
    # 设置参数
    np.set_printoptions(precision=30)
    pop = 30  # 种群数量
    MaxIter = 1000  # 最大迭代次数
    Maxstep = 1  # 每运行30次取最小
    dim = 30  # 维度
    lb = -100 * np.ones([dim, 1])  # 下边界
    ub = 100 * np.ones([dim, 1])  # 上边界
    # 适应度函数选择
    fobj = fun5
    # pdb.set_trace()
    # LSO
    LSO_GbestScore, LSO_GbestPositon, LSO_Curve = LSO(pop, dim, lb, ub, MaxIter, Maxstep, fobj)
    print('LSO最优适应度值：', LSO_GbestScore)
    print('LSO最优解：', LSO_GbestPositon)
    










