import numpy as np
from matplotlib import pyplot as plt
import LSO_1
import CLSO_1
import DLSO
import pdb
import PSO_1
import GWO_1
import CLSO_2
from logger import logger
import time
import fun_test



'''主函数 '''
#设置狮群算法参数
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
#适应度函数选择
ls = [fun1, fun2, fun3, fun4, fun5, fun6, fun7, fun8, fun9, fun10, fun11, fun12, fun13]
# ls = [fun12]
for idx, fobj in enumerate(ls):
    # logger.info('当前测试函数为：%s' % fobj.__name__)
    print('当前测试函数为：%s' % fobj.__name__)
    np.set_printoptions(precision=30)
    pop = 30 #种群数量
    MaxIter = 100 #最大迭代次数
    dim = 2  #维度
    lb, ub = func_lb_ub_map.get(fobj,[-5, 5])
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


    # # PSO
    # pso = PSO_1.PSO(dim, pop, MaxIter, x_max, v_max, tol=0, C1=C1, C2=C2, W=W, fun=fobj)
    # fit_var_list, best_pos = pso.update_ndim()
    # # print("最优位置:" + str(best_pos))
    # # print("PSO最优解:" + str(fit_var_list[-1]))
    # logger.info("PSO最优解:{}".format(str(fit_var_list[-1])))
    #
    #
    # # GWO
    # GWO_Curve, GWO_GbestScore = GWO_1.GWO(fobj, lb1, ub1, dim, pop, MaxIter)
    # # print('GWO最优适应度值：', GWO_GbestScore)
    # logger.info('GWO最优适应度值：{}'.format(GWO_GbestScore))

    # LSO
    LSO_GbestScore,LSO_GbestPositon,LSO_Curve = LSO_1.LSO(pop,dim,lb,ub,MaxIter,fobj,gif=True)
    print('LSO最优适应度值：',LSO_GbestScore)
    # logger.info('LSO最优解：{}'.format(LSO_GbestScore))

    # # CLSO
    # CLSO_GbestScore,CLSO_GbestPositon,CLSO_Curve = CLSO_1.CLSO(pop,dim,lb,ub,MaxIter,fobj)
    # # print('CLSO最优适应度值：',CLSO_GbestScore[0])
    # logger.info('CLSO最优解：{}'.format(CLSO_GbestScore))
    # # print('CLSO最优解：',CLSO_GbestPositon)

    # CLSO_2
    CLSO_2_GbestScore, CLSO_2_GbestPositon, CLSO_2_Curve = CLSO_2.CLSO(pop, dim, lb, ub, MaxIter, fobj,gif=True)
    print('CLSO_2最优适应度值：', CLSO_2_GbestScore)
#     logger.info('CLSO_2最优解：{}'.format(CLSO_2_GbestScore))
    # 绘制适应度曲线
    plt.figure(idx+1)
    # plt.subplot(4,4,idx+1)
    # plt.plot(range(len(fit_var_list)), fit_var_list, alpha=0.5, ls='--',label='PSO')
    # plt.plot(range(MaxIter), np.array(GWO_Curve), color='y', ls='-', label='GWO')
    plt.plot(range(len(LSO_Curve)), np.array(LSO_Curve), color = 'g', ls=':',label='LSO')
    # plt.plot(range(len(CLSO_Curve)), np.array(CLSO_Curve), color = 'r', ls='',label='CLSO')
    plt.plot(range(MaxIter), np.array(CLSO_2_Curve), color='b', ls='-.', label='CLSO_2')
    # plt.semilogy(Curve,'r-',linewidth=2)  # semilogy(Y) 使用 y 轴的以 10 为基数的对数刻度和 x 轴的线性刻度创建一个绘图
    plt.xlabel('Iteration')
    plt.ylabel("Fitness")
    plt.grid(True,which='both')
    plt.yscale('log')
    # plt.yscale('symlog')
    plt.title('%s'%fobj.__name__,fontsize='large')
    plt.legend()
#     plt.savefig("pictures/test_function_picture/{}.png".format(str(int(time.time()))))
plt.show()