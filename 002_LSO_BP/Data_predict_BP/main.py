from CLSO_bp import CLSO
from CLSO_bp_main import BPNN, data_LSOToBP,data
from logger import logger
from BP_house import BP, FontProperties
import numpy as np
import matplotlib.pyplot as plt
from GLSO_bp_2 import GLSO
from CLSO_bp_main import mse, rmse, mae, TV, TSD
from GWO_1 import GWO
from LSO_1 import LSO

input_num = 13
hidden_num = 14
output_num = 1
pop = 50
dim = input_num * hidden_num + hidden_num * output_num + hidden_num + output_num
lb1 = -5
ub1 = 5
lb = np.array([lb1]*dim)
ub = np.array([ub1]*dim)
MaxIter = 50 #狮群迭代次数
Maxepoch = 1000
fun = BPNN
learning_rate = 0.005

# CLSO_BP
logger.info("狮群数量：{} 狮群迭代次数：{}".format(pop,MaxIter))
# _, GbestPositon, _ = CLSO(pop, dim, lb, ub, MaxIter,fun)
_, GLSO_GbestPositon, _ = GLSO(pop, dim, lb, ub, MaxIter, fun)
# logger.info(GbestPositon)
w,b = data_LSOToBP(GLSO_GbestPositon, input_num, hidden_num, output_num)
GLSO_MSE_loss = BPNN(data, w, b, epochs=Maxepoch, learning_rate=learning_rate, need_paint=True,need_loss_log=True)
# print('MSE_loss=',GLSO_MSE_loss)
logger.info('GLSO_MSE_loss={}'.format(GLSO_MSE_loss))

# LSO_BP
_, LSO_GbestPositon, _ = LSO(pop, dim, lb, ub, MaxIter, fun)
w,b = data_LSOToBP(LSO_GbestPositon, input_num, hidden_num, output_num)
GLSO_MSE_loss = BPNN(data, w, b, epochs=Maxepoch, learning_rate=learning_rate, need_paint=True,need_loss_log=True, optimization_algorithm='LSO_BP')
logger.info('LSO_MSE_loss={}'.format(GLSO_MSE_loss))

# # GWO_BP
# _, _, GWO_GbestPositon = GWO(fun, lb1, ub1, dim, pop, MaxIter)
# w, b = data_LSOToBP(GWO_GbestPositon, input_num, hidden_num, output_num)
# GWO_MSE_loss = BPNN(data, w, b, epochs=Maxepoch, learning_rate=learning_rate, need_paint=True, need_loss_log=True,optimization_algorithm='GWO_BP')
# print('GWO_MSE_loss=', GWO_MSE_loss)
#
#
# # BP
# test_set_input, test_set_output, predict,y_train_predict,y_train_real, l_arr = BP(data,max_epoch=Maxepoch, learning_rate=learning_rate)
# predict = predict.detach().numpy().reshape(-1,1)
# real = np.array(test_set_output).reshape(-1,1)
# logger.info('BP_real_data : {}'.format(real.reshape(1,-1)))
# logger.info('BP_predict_data : {}'.format(predict.reshape(1,-1)))
# logger.info('BP_train_Loss : ')
# mse(y_train_real,y_train_predict)
# rmse(y_train_real,y_train_predict)
# mae(y_train_real,y_train_predict)
# TV(y_train_real,y_train_predict)
# TSD(y_train_real,y_train_predict)
# logger.info('BP_test_Loss : ')
# mse(real, predict)
# rmse(real, predict)
# mae(real, predict)
# TV(real, predict)
# TSD(real, predict)
# # 图1
# plt.subplot(2, 2, 1)
# x = np.arange(Maxepoch)
# y = np.array(l_arr)
# myfont = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
# plt.plot(x, y, label="BP")
# plt.title('The loss value changes with the number of iterations', fontproperties=myfont)
# plt.xlabel('iterations times', fontproperties=myfont)
# plt.ylabel('Each iteration loses the average value of the function', fontproperties=myfont)
# plt.legend()



# 图2
# plt.subplot(2, 2, 3)
# x = np.arange(test_set_input.shape[0])
# y1 = np.array(predict.detach().numpy())
# y2 = np.array(test_set_output)
# myfont = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
# plt.plot(x, y1, c='red', label='predict_data')
# plt.plot(x, y2, c='yellow', label='real_data')
# plt.legend()
# plt.title('BP下预测值与实际值', fontproperties=myfont)
# plt.xlabel('数据/组', fontproperties=myfont)
# plt.ylabel('房价', fontproperties=myfont)
plt.show()
