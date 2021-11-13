# coding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, \
    r2_score
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def LoadData(path):
    return pd.read_csv(path, engine='python')


def Data_Preprocess(path, show_corr_indx=False, step_sampling=True):
    '''
    path : 数据集路径
    show_corr_indx: 是否显示相关性
    step_sampling:  是否分层抽样
    '''
    # 一、查看数据相关信息
    # 1.1 加载数据集 [20640,10]
    # path = u'加州房价数据集.csv'
    # show_corr_indx = True  # 是否显示相关性
    # step_sampling = True  # 是否分层抽样
    data = LoadData(path)
    data.head(5)
    data.info(verbose=None)
    # data.hist(bins=50,figsize=(20,15),edgecolor="black")  出数据分布直方图

    # 1.2 划分数据集 训练集80%(16512)  测试集20%(4128)
    if not step_sampling:
        house_train, house_test = train_test_split(data, test_size=0.2, random_state=24)
    else:
        data["income_cat"] = pd.cut(data["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
        # 根据income_cat数据绘制直方图
        # data["income_cat"].hist()
        s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in s.split(data, data["income_cat"]):
            # train_index: [17606 18632 14650 ... 13908 11159 15775]
            start_train_set = data.loc[train_index]
            # test_index: [5241 10970 20351 ...  4019 12107  2398]
            start_test_set = data.loc[test_index]
        # 删除‘incom_cat’列
        for dataset in (start_train_set, start_test_set):
            dataset.drop("income_cat", axis=1, inplace=True)
        house_train = start_train_set.copy()
        house_test = start_test_set.copy()
    ''' 计算每个属性之间的标准相关系数（也称作皮尔逊相关系数）,
    查看每个属性和median_house_value的相关系数，数值在[-1,1]之间。
    接近1时，说明两个特征有很明显的正相关关系;
    相反则是有很明显的负相关关系;如果是0，则说明没有相关关系。'''

    # 1.3 计算相关系数
    if show_corr_indx:
        housing = house_train.copy()
        corr_matrix = housing.corr()
        corr_matrix["median_house_value"].sort_values(ascending=False)
        # 查看heatmap热度图，两个特征之间的相互关系
        fig, ax = plt.subplots(figsize=(15, 10))
        # sns.set(font_scale=1.25)
        sns.heatmap(corr_matrix, fmt='.2f', cmap='GnBu_r', square=True, linewidths=0.3, annot=True)
        plt.show()

    # 二、数据预处理

    # 训练集
    # 2.1 拆分数据,获得特征矩阵H以及只含有median_house_value作为真实的label值，y值
    housing = house_train.drop("median_house_value", axis=1)
    housing_labels = house_train["median_house_value"].copy()
    housing_copy = housing.drop("ocean_proximity", axis=1)  # 去掉干扰值
    num_attribs = list(housing_copy)
    cat_attribs = ['ocean_proximity']
    # 2.2 缺失值用中位数填充，字符型使用one-hot编码
    # ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    num_type_pipeline = Pipeline([
        ('simputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())

    ])
    full_pipeline = ColumnTransformer([
        ("num", num_type_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    train_data = full_pipeline.fit_transform(housing)  # x值
    train_label = housing_labels.values

    # 测试集
    housing_test = house_test.drop("median_house_value", axis=1)
    housing_test_labels = house_test["median_house_value"].copy()
    housing_test_copy = housing_test.drop("ocean_proximity", axis=1)  # 去掉干扰值
    num_test_attribs = list(housing_test_copy)
    cat_test_attribs = ['ocean_proximity']
    num_test_pipeline = Pipeline([
        ('simputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])
    full_test_pipeline = ColumnTransformer([
        ("num", num_test_pipeline, num_test_attribs),
        ("cat", OneHotEncoder(), cat_test_attribs),
    ])
    test_data = full_test_pipeline.fit_transform(housing_test)
    test_label = housing_test_labels.values

    # #评价模型
    # GBR_mse = mean_squared_error(housing_labels,GBR_housing_predictions)
    # GBR_ab = mean_absolute_error(housing_labels,GBR_housing_predictions)
    # GBR_va = explained_variance_score(housing_labels,GBR_housing_predictions)
    # GBR_mab = median_absolute_error(housing_labels,GBR_housing_predictions)
    # GBR_r2 = r2_score(housing_labels,GBR_housing_predictions)
    # GBR_rmse = np.sqrt(GBR_mse)

    # #输出评价结果
    # print('GBR回归模型的平均值绝对值为：',GBR_ab)
    # print('GBR回归模型的中值绝对值为：',GBR_mab)
    # print('GBR回归模型的可解释方差值为：',GBR_va)
    # print('GBR回归的决定系数为：',GBR_r2)
    # print ('GBR回归模型的标准差为：',GBR_rmse)

    return train_data, train_label, test_data, test_label

#
# path = u'加州房价数据集.csv'
# train_data, train_label, test_data, test_label = Data_Preprocess(path)
# print(train_data.shape)
