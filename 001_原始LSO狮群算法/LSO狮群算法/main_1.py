import numpy as np
from matplotlib import pyplot as plt
import LSO_1
import CLSO_1
import DLSO
import pdb
import PSO_1
import GWO_1
import CLSO_2

'''定义目标函数用户可选fun1 - fun6 , 也可以自己定义自己的目标函数'''
def preprocess_X(X):
    if len(X) == 1:
        X = X[0]
    return X

def fun1(X):  # Sphere Function Xi=0 f(x)=0  单峰函数，可用来考验算法的收敛速度
    X = preprocess_X(X)
    O=np.sum(X*X + X*X)
    return O

def fun2(X):  #  SchwefelProblem  np.prod()函数用来计算所有元素的乘积, Xi=0 f(x)=0
    # 高峰单维函数，考验算法收敛速度和收敛精度
    X = preprocess_X(X)
    O=np.sum(np.abs(X))+np.prod(np.abs(X))
    # print(X.shape,O.shape)
    return O

def fun3(X):
    X = preprocess_X(X)
    O=0
    for i in range(len(X)):
        O=O+np.square(np.sum(X[0:i+1]))
    return O

def fun4(X):
    X = preprocess_X(X)
    O=np.max(np.abs(X))
    return O

def fun5(X):  # Generalized Rosenbrock  Xi=1, f(x)=0
    # 该函数全局最优点位于一个平滑、狭长的抛物线形山谷内，由于函数为优化算法提供的信息比较有限，使算法很难辨别搜索方向，查找最优解也变得十分困难
    X = preprocess_X(X)
    X_len = len(X)
    O = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1))
    return O

def fun6(X):
    X_len = len(X)
    O1 = 0
    for i in range(0, X_len-1):
        O1 += i * np.power(X[i],4)
    O = O1 + np.random.rand()
    return O

def fun7(X):
    X = preprocess_X(X)
    O = np.sum(X * X - 10 * np.cos(2 * np.pi * X) + 10)
    return O

def fun8(X):  # Ackley Xi=0 f(x)=0
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
    X = preprocess_X(X)
    O1 = -20 * np.exp(-0.2 * np.sqrt(1 / dim * np.sum(X * X))) + 20 + np.e
    O2 = 0
    for i in range(len(X)):
        O2 = O2 + np.cos(2 * np.pi * X[i])
    O = O1 - np.exp(1 / dim * O2)
    # O = -20 * np.exp(-0.2 * np.sqrt(1/dim * np.sum(X * X))) + 20 + np.e - np.exp(-0.2 * np.sqrt(1/dim * np.sum(np.cos(2 * np.pi * X))))
    return O
# def fun6(X):  # Step Xi=0 f(x)=0  单峰函数，可用来考验算法的收敛速度
#     X = preprocess_X(X)
#     O=np.sum(np.square(np.abs(X+0.5)))
#     return O


# def fun7(X): # Generalized Schwefel‘s  30维时理论最小值f(420.9687) = -12569.5  |Xi|<=500，2维时,理论最小值f( 420.96875, 420.96875 ) = -837.965759277
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
#     X = preprocess_X(X)
#     O = -np.sum(X * np.sin(np.sqrt(np.abs(X))))
#     return O

# def fun8(X):  # Rastrigin 全局最小值为所有变量均为0时取得O=0 Xi=0 f(x)=0
#     # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
#     X = preprocess_X(X)
#     O = 10 * dim + np.sum(X*X - 10*np.cos(2*np.pi*X))
#     return O

def fun9(X):  # Generalized Griewank function 全局最小f(x)=0,x=(0,...,0)
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
    X = preprocess_X(X)
    X_len = len(X)
    O1 = 1
    for i in range(1,X_len):
        O1 = O1 * np.cos(X[i]/np.sqrt(i))
    O = 1/4000 * np.sum(X*X) - O1 + 1
    return O
# #
# def fun10(X):
#     X = preprocess_X(X)
#     O = np.square(X[1] - 5.1 / (4*np.pi*np.pi) * np.square(X[0]) + 5/np.pi * X[0] - 6) + 10 * (1 - 1/(8*np.pi) * np.cos(X[0])) + 10
#     return O

# “基于改进狮群算法和BP神经网络模型的房价预测”中的测试函数
def fun10(X):  #  weierstrass
    X = preprocess_X(X)
    k_max = 20
    a = 0.5
    b = 3
    O1 = 0
    O2 = 0
    X_len = len(X)
    for i in range(X_len):
        for j in range(k_max+1):
            O1 = O1 + np.power(a,j) * np.cos((2*np.pi*np.power(b,j)*(X[i]+0.5)))
    for k in range(k_max+1):
        O2 = O2 + np.power(a,k) * np.cos(np.pi*np.power(b,k))

    O = O1 - dim * O2

    return O

def fun11(X):  # HOLDER TABLE FUNCTION F(x) = -19.2085 X:[-10,10],四个全局最小点
    X = preprocess_X(X)
    O = -1 * np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0]*X[0] + X[1]*X[1])/np.pi)))
    return O

def fun12(X): # LEVY FUNCTION N. 13   f(1,1) = 0
    X = preprocess_X(X)
    O = np.square(np.sin(3 * np.pi * X[0])) + np.square(X[0]-1)*(1 + np.square(np.sin(3*np.pi*X[1]))) + np.square(X[1]-1)*(1 + np.square(np.sin(2*np.pi*X[1])))
    return O

def fun13(X): # SCHAFFER FUNCTION N. 2
    X = preprocess_X(X)
    O = 0.5 + (np.square(np.sin(X[0]*X[0] - X[1]*X[1])) - 0.5) / np.square(1 + 0.001*(X[0]*X[0] + X[1]*X[1]))

    return O
def fun14(X):  # Step Xi=0 f(x)=0  单峰函数，可用来考验算法的收敛速度
    X = preprocess_X(X)
    O=np.sum(np.square(np.abs(X+0.5)))
    return O


def fun15(X): # Generalized Schwefel‘s  30维时理论最小值f(420.9687) = -12569.5  |Xi|<=500，2维时,理论最小值f( 420.96875, 420.96875 ) = -837.965759277
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
    X = preprocess_X(X)
    O = -np.sum(X * np.sin(np.sqrt(np.abs(X))))
    return O

def fun16(X):  # Rastrigin 全局最小值为所有变量均为0时取得O=0 Xi=0 f(x)=0
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
    X = preprocess_X(X)
    O = 10 * dim + np.sum(X*X - 10*np.cos(2*np.pi*X))
    return O
'''主函数 '''
#设置狮群算法参数
if __name__ == '__main__':
    func_lb_ub_map = {
        fun1: [-100, 100],
        fun2: [-10, 10],
        fun3: [-100, 100],
        fun4: [-100, 100],
        fun5: [-30, 30],
        fun6: [-1.28, 1.28],
        fun7: [-5.12, 5.12],
        fun8: [-32, 32],
        fun9: [-600, 600],
        fun10: [-5, 5],
        fun11: [-10, 10],
        fun12: [-10, 10],
        fun13: [-100, 100]
    }
    Maxstep = 30  # 每运行Maxstep次取最小
    #适应度函数选择
    # ls = [fun1, fun2, fun3, fun4, fun5, fun6, fun7, fun8, fun9, fun10, fun11, fun12, fun13]
    ls = [fun16]
    for idx, fobj in enumerate(ls):
        print('当前测试函数为：%s' % fobj.__name__)
        np.set_printoptions(precision=30)
        pop = 30 #种群数量
        MaxIter = 1000 #最大迭代次数
        Maxstep = 1  # 每运行Maxstep次取最小
        dim = 30  #维度
        lb, ub = func_lb_ub_map.get(fobj,[-100, 100])
        # 设置灰狼算法参数
        lb1 = lb
        ub1 = ub
        # 狮群
        lb = lb*np.ones([dim, 1]) #下边界
        ub = ub*np.ones([dim, 1])#上边界
        # 设置粒子群算法参数def __init__(self, dim, size, iter_num, x_max, max_vel, tol, best_fitness_value=float('Inf'), C1=2, C2=2, W=1):
        x_max = ub1
        v_max = 0.5
        C1 = 0.49445
        C2 = 1.49445
        W = 1


        # PSO
        pso = PSO_1.PSO(dim, pop, MaxIter, x_max, v_max, 0, C1, C2, W, fun=fobj)
        fit_var_list, best_pos = pso.update_ndim()
        # print("最优位置:" + str(best_pos))
        print("PSO最优解:" + str(fit_var_list[-1]))


        # GWO
        GWO_Curve, GWO_GbestScore = GWO_1.GWO(fobj, lb1, ub1, dim, pop, MaxIter)
        print('GWO最优适应度值：', GWO_GbestScore)

        # LSO
        LSO_GbestScore,LSO_GbestPositon,LSO_Curve = LSO_1.LSO(pop,dim,lb,ub,MaxIter,fobj)
        print('LSO最优适应度值：',LSO_GbestScore)
        # print('LSO最优解：',LSO_GbestPositon)

        # CLSO
        CLSO_GbestScore,CLSO_GbestPositon,CLSO_Curve = CLSO_1.CLSO(pop,dim,lb,ub,MaxIter,fobj)
        print('CLSO最优适应度值：',CLSO_GbestScore)
        # print('CLSO最优解：',CLSO_GbestPositon)

        # GLSO
        GLSO_GbestScore, GLSO_GbestPositon, GLSO_Curve = CLSO_2.CLSO(pop, dim, lb, ub, MaxIter, fobj)
        print('CLSO最优适应度值：', GLSO_GbestScore)
        #绘制适应度曲线
        plt.figure(idx+1)
        plt.plot(range(len(fit_var_list)), fit_var_list, alpha=0.5, ls='--',label='PSO')
        plt.plot(range(len(LSO_Curve)), np.array(LSO_Curve), color = 'g', ls=':',label='LSO')
        plt.plot(range(len(CLSO_Curve)), np.array(CLSO_Curve), color = 'r', ls='-.',label='CLSO')
        plt.plot(range(len(GLSO_Curve)), np.array(GLSO_Curve), color='b', ls='-', label='GLSO')
        plt.plot(range(MaxIter), np.array(GWO_Curve), color='y', ls='-',label = 'GWO')
        # plt.semilogy(Curve,'r-',linewidth=2)  # semilogy(Y) 使用 y 轴的以 10 为基数的对数刻度和 x 轴的线性刻度创建一个绘图
        plt.xlabel('Iteration',fontsize='medium')
        plt.ylabel("Fitness",fontsize='medium')
        plt.grid(True,which='both')
        plt.yscale('log')
        # plt.yscale('symlog')
        #plt.ylim(0,10)
        plt.title('%s'%fobj.__name__,fontsize='large')
        plt.legend()
    plt.show()