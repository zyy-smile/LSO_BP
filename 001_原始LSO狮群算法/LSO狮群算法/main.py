import numpy as np
from matplotlib import pyplot as plt
import LSO
import Chaos_LSO
import DLSO
import PSO_1
import pdb
'''定义目标函数用户可选fun1 - fun6 , 也可以自己定义自己的目标函数'''
def fun1(X):  # Sphere Function Xi=0 f(x)=0  单峰函数，可用来考验算法的收敛速度
        O=np.sum(X*X + X*X)
        return O

def fun2(X):  #  SchwefelProblem  np.prod()函数用来计算所有元素的乘积, Xi=0 f(x)=0
    # 高峰单维函数，考验算法收敛速度和收敛精度
    O=np.sum(np.abs(X))+np.prod(np.abs(X))
    # print(X.shape,O.shape)
    return O

def fun3(X):
    O=0
    for i in range(len(X)):
        O=O+np.square(np.sum(X[0:i+1]))   
    return O

def fun4(X):
    O=np.max(np.abs(X))
    return O

def fun5(X):  # Generalized Rosenbrock  Xi=1, f(x)=0
    # 该函数全局最优点位于一个平滑、狭长的抛物线形山谷内，由于函数为优化算法提供的信息比较有限，使算法很难辨别搜索方向，查找最优解也变得十分困难
    X_len = len(X)
    O = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1))
    return O

def fun6(X):  # Step Xi=0 f(x)=0  单峰函数，可用来考验算法的收敛速度
    O=np.sum(np.square(np.abs(X+0.5)))
    return O


def fun7(X): # Generalized Schwefel‘s  30维时理论最小值f(420.9687) = -12569.5  |Xi|<=500，2维时,理论最小值f( 420.96875, 420.96875 ) = -837.965759277
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
    O = -np.sum(X * np.sin(np.sqrt(np.abs(X))))
    return O

def fun8(X):  # Rastrigin 全局最小值为所有变量均为0时取得O=0 Xi=0 f(x)=0
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
    O = 10 * dim + np.sum(X*X - 10*np.cos(2*np.pi*X))
    return O

def fun9(X):  # Generalized Griewank function 全局最小f(x)=0,x=(0,...,0)
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
    X_len = len(X)
    O1 = 1
    for i in range(1,X_len):
        O1 = O1 * np.cos(X[i]/np.sqrt(i))
    O = 1/4000 * np.sum(X*X) - O1 + 1
    return O

#“基于改进狮群算法和BP神经网络模型的房价预测”中的测试函数
def fun10(X):  #  weierstrass
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

def fun11(X): #Ackley Xi=0 f(x)=0
    # 多峰函数，局部最优点的数量随维度指数递增，可有效检验算法的全局搜索性能
    O1 = -20 * np.exp(-0.2 * np.sqrt(1/dim * np.sum(X * X))) + 20 + np.e
    O2 = 0
    for i in range(len(X)):
        O2 = O2 + np.cos(2 * np.pi * X[i])
    O =  O1 - np.exp(1/dim * O2)
    # O = -20 * np.exp(-0.2 * np.sqrt(1/dim * np.sum(X * X))) + 20 + np.e - np.exp(-0.2 * np.sqrt(1/dim * np.sum(np.cos(2 * np.pi * X))))
    return O

def fun12(X): # LEVY FUNCTION N. 13   f(1,1) = 0
    O = np.square(np.sin(3 * np.pi * X[0])) + np.square(X[0]-1)*(1 + np.square(np.sin(3*np.pi*X[1]))) + np.square(X[1]-1)*(1 + np.square(np.sin(2*np.pi*X[1])))
    return O

def fun13(X): # SCHAFFER FUNCTION N. 2
    O = 0.5 + (np.square(np.sin(X[0]*X[0] - X[1]*X[1])) - 0.5) / np.square(1 + 0.001*(X[0]*X[0] + X[1]*X[1]))

    return O
'''主函数 '''
#设置参数
np.set_printoptions(precision=30)
pop = 30 #种群数量
MaxIter = 10 #最大迭代次数
Maxstep = 1  # 每运行30次取最小
dim = 10  #维度
lb = -100*np.ones([dim, 1]) #下边界
ub = 100*np.ones([dim, 1])#上边界
#适应度函数选择
fobj = fun13
# pdb.set_trace()
# LSO
LSO_GbestScore,LSO_GbestPositon,LSO_Curve = LSO.LSO(pop,dim,lb,ub,MaxIter,Maxstep,fobj)
print('LSO最优适应度值：',LSO_GbestScore)
print('LSO最优解：',LSO_GbestPositon)
# Chao_LSO
# CLSO_GbestScore,CLSO_GbestPositon,CLSO_GbestScore_max,CLSO_GbestPositon_max,CLSO_GbestScore_mean,CLSO_GbestPositon_mean,CLSO_Curve = Chaos_LSO.Chao_LSO(pop,dim,lb,ub,MaxIter,Maxstep,fobj)
# CLSO_GbestScore,CLSO_GbestPositon,CLSO_Curve = Chaos_LSO.Chao_LSO(pop,dim,lb,ub,MaxIter,Maxstep,fobj)
# print('CLSO最优适应度值：',CLSO_GbestScore)
# print('CLSO最优解：',CLSO_GbestPositon)

# DLSO
DLSO_GbestScore,DLSO_GbestPositon,DLSO_Curve = DLSO.DLSO(pop,dim,lb,ub,MaxIter,Maxstep,fobj)
print('DLSO最优适应度值：',DLSO_GbestScore)
print('DLSO最优解：',DLSO_GbestPositon)
# print('运行30次CLSO平均适应度值：',CLSO_GbestScore_mean)
# print('CLSO最优解：',CLSO_GbestPositon_mean)
# print('运行30次CLSO最差适应度值：',CLSO_GbestScore_max)
# print('运行30次CLSO最差解：',CLSO_GbestPositon_max)
#绘制适应度曲线
plt.figure(1)
plt.plot(range(len(LSO_Curve)), np.array(LSO_Curve), color = 'g',label='LSO')
# plt.plot(range(len(CLSO_Curve)), np.array(CLSO_Curve), color = 'r',label='CLSO')
plt.plot(range(len(DLSO_Curve)), np.array(DLSO_Curve), color = 'y',label='DLSO')
# plt.semilogy(Curve,'r-',linewidth=2)  # semilogy(Y) 使用 y 轴的以 10 为基数的对数刻度和 x 轴的线性刻度创建一个绘图
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid(True,which='both')
#plt.yscale('symlog')
#plt.ylim(0,10)
plt.title('The diffience between LSO and CLSO',fontsize='large')
plt.legend()
plt.show()
