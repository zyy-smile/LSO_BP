import random
import numpy
import pdb
from CLSO_bp_main import BPNN, data_LSOToBP, data, data_BPToLSO
from CLSO_bp import cal_fitness, CaculateFitness
import numpy as np

input_num = 13
hidden_num = 14
output_num = 1
def cal_fitness(x, fun, order=0, fitness=None):
    w, b = data_LSOToBP(x, input_num, hidden_num, output_num)
    res = fun(data, w, b, epochs=100, learning_rate=0.01)
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

def GWO(fun, lb, ub, dim, SearchAgents_no, Max_iter):


    # 初始化 alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)  # 位置.形成30的列表
    Alpha_score = float("inf")  # 这个是表示“正负无穷”,所有数都比 +inf 小；正无穷：float("inf"); 负无穷：float("-inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")  # float() 函数用于将整数和字符串转换成浮点数。

    # list列表类型
    if not isinstance(lb, list):  # 作用：来判断一个对象是否是一个已知的类型。 其第一个参数（object）为对象，第二个参数（type）为类型名，若对象的类型与参数二的类型相同则返回True
        lb = [lb for i in range(dim)]   # 生成[100，100，.....100]30个
    if not isinstance(ub, list):
        ub = [ub for i in range(dim)]

    # Initialize the positions of search agents初始化所有狼的位置
    Positions = numpy.zeros((SearchAgents_no, dim)) # 狼群
    # 形成（SearchAgents_no * dim）个数[-100，100)以内
    for i in range(SearchAgents_no):
        for j in range(dim):
            Positions[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
    Convergence_curve = numpy.zeros(Max_iter)




    #迭代寻优
    for l in range(Max_iter):  # 迭代1000
        for i in range(0, SearchAgents_no):  # 30
            # 返回超出搜索空间边界的搜索代理

            for j in range(dim):  # 30
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])  # clip这个函数将将数组中的元素限制在a_min(-100), a_max(100)之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。

            # 计算每个搜索代理的目标函数
            # fitness = objf(Positions[i, :])  # 把某行数据带入函数计算
            fitness = cal_fitness(Positions[i, :], fun)
            # print("经过计算得到：",fitness)

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        # 以上的循环里，Alpha、Beta、Delta

        a = 2 - l * ((2) / Max_iter);  #   a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  # Equation (3.3)
                C1 = 2 * r2;  # Equation (3.4)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[
                    i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;

                Positions[i, j] = (X1 + X2 + X3) / 3  # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve[l] = Alpha_score;

        # if (l % 10 == 0):
        #     print(['迭代次数为' + str(l) + ' 的迭代结果' + str(Alpha_score)]);  # 每一次的迭代结果
    # pdb.set_trace()
    # Convergence_curve = Convergence_curve.reshape((1,-1))
    GWO_best = Convergence_curve[-1]
    return Convergence_curve, GWO_best, Alpha_pos


# if __name__ == '__main__':
#     input_num = 13
#     hidden_num = 14
#     output_num = 1
#     pop = 50
#     dim = input_num * hidden_num + hidden_num * output_num + hidden_num + output_num
#     lb = -5
#     ub = 5
#     MaxIter = 10 # 狮群迭代次数
#     Maxepoch = 200
#     fun = BPNN
#     learning_rate = 0.005
#     _, _, GbestPositon  = GWO(fun, lb, ub, dim, pop, MaxIter)
#     w, b = data_LSOToBP(GbestPositon, input_num, hidden_num, output_num)
#     MSE_loss = BPNN(data, w, b, epochs=Maxepoch, learning_rate=learning_rate, need_paint=True, need_loss_log=True,optimization_algorithm='GWO_BP')
#     print('MSE_loss=', MSE_loss)
# # # #主程序
# # func_details = ['F1', -100, 100, 30]
# function_name = func_details[0]
# Max_iter = 1000#迭代次数
# lb = -100#下界
# ub = 100#上届
# dim = 30#狼的寻值范围
# SearchAgents_no = 5#寻值的狼的数量
# x = GWO(F1, lb, ub, dim, SearchAgents_no, Max_iter)

