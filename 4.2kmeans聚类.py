# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import decomposition as skldec #用于主成分分析降维的包

##df=  pd.read_csv("30_school_sim.csv",index_col = 0,error_bad_lines=False)
df=  pd.read_csv(r'E:\\LearningMaterial\\shujuwajue\\first\\test_sim_7wan.csv',index_col=0,error_bad_lines=False)
#print(spott.iloc[1])
# print(spot)
K=5
estimator = KMeans(K)#构造聚类器，构造一个聚类数为K的聚类器

estimator.fit(df)#聚类

label_pred = estimator.labels_ #获取聚类标签

print('聚类标签',label_pred)

centroids = estimator.cluster_centers_ #获取聚类中心

print('聚类中心',centroids)

inertia = estimator.inertia_ # 获取聚类准则的总和

print('聚类准则的总和',inertia)

'''
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
#这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
color = 0
j = 0

for i in label_pred:

    plt.plot([df.iloc[j:j+1,0]], [df.iloc[j:j+1,1]], mark[i], markersize = 5)
    j +=1
    
plt.show()

'''
#根据两个最大的主成分进行绘图
pca = skldec.PCA(n_components = 0.95)    #选择方差95%的占比
pca.fit(df)   #主城分析时每一行是一个输入数据
result = pca.transform(df)  #计算结果
plt.figure()  #新建一张图进行绘制
plt.scatter(result[:, 0], result[:, 1], c=label_pred, edgecolor='k') #绘制两个主成分组成坐标的散点图
'''
for i in range(result[:,0].size):
    plt.text(result[i,0],result[i,1],df.index[i])     #在每个点边上绘制数据名称
'''
x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0]*100.0),2)   #x轴标签字符串
y_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[1]*100.0),2)   #y轴标签字符串
plt.xlabel(x_label)    #绘制x轴标签
plt.ylabel(y_label)    #绘制y轴标签

plt.show()

