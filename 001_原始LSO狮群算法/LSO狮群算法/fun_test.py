import numpy as np
import matplotlib.pyplot as plt
dim = 2

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
# 动态二维图
def dynamic_graph(populations, best_fitness):
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
    plt.draw()
    plt.pause(0.1)
    plt.ioff()
    # plt.close()
    plt.clf()
    # plt.show()
    # 设置狮群算法参数

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
# 适应度函数选择
ls = [fun1, fun2, fun3, fun4, fun5, fun6, fun7, fun8, fun9, fun10, fun11, fun12, fun13]
