import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 测试函数
# 1、Ackley’s function
def fun1(z_min = 0,z_max = 15, offset = 0):
    x, y = get_x_and_y(-5,5,-5,5)
    z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 20 + np.e
    return x, y, z, 'Ackely function', z_min, z_max, offset

# 2、Sphere function
def fun2(z_min = 0,z_max = 20,offset = 0):
    x,y = get_x_and_y( -3,3,-3,3)
    z = x ** 2 + y ** 2
    return x,y,z, "Sphere function", z_min, z_max, offset

# 3、Rosebrock function
def fun3(z_min = 0,z_max = 1500,offset = 0):
    x, y = get_x_and_y(-1, 1, -1, 1)
    z = 100 * ((y - x ** 2) ** 2 + (x -1) ** 2)
    return x,y,z, "Rosebrock function", z_min, z_max, offset

# 4、Beale’s function
def fun4(z_min = 0,z_max = 2000,offset=0):
    x, y = get_x_and_y(-4.5, 4.5, -4.5, 4.5)
    z = (1.5 - x + x * y) ** 2 +  (2.25 - x + x * y) ** 2 + (2.625 -x + x * y) ** 2
    return x, y, z, "Beale function", z_min, z_max, offset

# 5、GoldsteinPrice function
def fun5(z_min =0 ,z_max =1000000 ,offset=0):
    x, y = get_x_and_y(-2,2,-2,2)
    z = (1 + ((x + y + 1) ** 2) * (19 - 14 * x + 3 * (x ** 2) - 14 * y + 6 * x * y + 3 * (y **2))) * (30 + ((2 * x - 3 * y) ** 2) * (18 - 32 * x + 12 * (x ** 2) + 48 * y - 36 * x * y + 27 * (y ** 2)))
    return x, y, z, "GoldsteinPrice function", z_min, z_max, offset

# 6、Booth’s function
def fun6(z_min=0, z_max=2500, offset=0):
    x, y = get_x_and_y(-10,10,-10,10)
    z = (x + 2 * y -7) ** 2 + (2 * x + y - 5) ** 2
    return x, y, z, "Booth function", z_min, z_max, offset

# 7、Bukin function
def  fun7(z_min =0 ,z_max =250 ,offset=0):
    x, y = get_x_and_y(-15,-5,-3,3)
    z = 100 * np.sqrt(np.fabs(y - 0.01 * (x ** 2))) + 0.01 * np.fabs(x + 10)
    return x, y, z, "Bukin function", z_min, z_max, offset

# 8、Matyas function
def fun8(z_min =0 ,z_max =100 ,offset=0):
    x, y = get_x_and_y(-10,10,-10,10)
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return x, y, z, "Matyas function", z_min, z_max, offset

# 9、Levi function
def fun9(z_min =0 ,z_max =450 ,offset=0):
    x, y = get_x_and_y(-10,10,-10,10)
    z = np.sin(3 * np.pi * x) ** 2 + ((x - 1) ** 2) * (1 + np.sin(3 * np.pi * y) ** 2) + ((y - 1) ** 2) * (1 + np.sin(2 * np.pi * y) ** 2)
    return x, y, z, "Levi function", z_min, z_max, offset

# 10、Three-hump camel function
def fun10(z_min =0 ,z_max =2000 ,offset=0):
    x, y = get_x_and_y(-5,5,-5,5)
    z = 2 * (x ** 2) - 1.05 * (x ** 4) + (x ** 6) / 6 + x * y + (y ** 2)
    return x, y, z, "Three-hump camel function", z_min, z_max, offset

# 11、Eason function
def fun11(z_min =-1.5 ,z_max =0 ,offset=-1.5):
    x, y = get_x_and_y(-10,10,-10,10)
    z = -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))
    return x, y, z, "Eason function", z_min, z_max, offset

# 12、Cross-in-tray function
def fun12(z_min =-4 ,z_max =0 ,offset=-4):
    x, y = get_x_and_y(-10,10,-10,10)
    z = -0.0001 * (np.fabs(np.sin(x) * np.sin(y) * np.exp(np.fabs(100 - (np.sqrt((x ** 2) + (y ** 2)) / np.pi)))) + 1) ** 0.1
    return x, y, z, "Cross-in-tray function", z_min, z_max, offset

# 13、Holder table function
def fun13(z_min =-30 ,z_max =0 ,offset=-30):
    x, y = get_x_and_y(-10,10,-10,10)
    z = -np.fabs(np.sin(x) * np.cos(y) * np.exp(np.fabs(1 - (np.sqrt(x ** 2 + y ** 2))/np.pi)))
    return x, y, z, "Holder table function", z_min, z_max, offset

# 14、 McCormick function
def fun14(z_min =0 ,z_max =50 ,offset=0):
    x, y = get_x_and_y(-1.5,4,-3,4)
    z = np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1
    return x, y, z, "McCormick function", z_min, z_max, offset

# 15、StyblinskiTang function
def fun15(z_min =-200 ,z_max =250 ,offset=-200):
    x, y = get_x_and_y(-5,5,-5,5)
    z1 = x ** 4 -16 * (x ** 2) + 5 * x
    z2 = y ** 4 -16 * (y ** 2) + 5 * y
    z = (z1 + z2) / 2
    return x, y, z, "StyblinskiTang function", z_min, z_max, offset



# 获取X=>[x,y]的值
def get_x_and_y(x_min, x_max, y_min, y_max):
    x = np.arange(x_min, x_max, 0.1)
    y = np.arange(y_min, y_max, 0.1)
    x, y = np.meshgrid(x, y)  # 生成网格点坐标矩阵
    return x, y

# 画图部分(X为二维时的图像)
def draw_pic_3D(x, y, z, title, z_min, z_max, offset):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap = plt.get_cmap('rainbow'),color='orangered')
    # 绘制等高线
    ax.contour(x,y,z,offset=offset,colors='green') # offset=offset:将图形映射到z=offset的平面上
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f (x1,x2)')
    plt.savefig("image")
    plt.show()

if __name__ == '__main__':
    x, y, z, title, z_min, z_max, offset = fun10()
    draw_pic_3D(x, y, z, title, z_min, z_max, offset)
