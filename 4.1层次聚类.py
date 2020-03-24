# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns  #用于绘制热图的工具包
from scipy.cluster import hierarchy  #用于进行层次聚类，画层次聚类图的工具包
from scipy import cluster   
import matplotlib.pyplot as plt
import pylab
from sklearn import decomposition as skldec #用于主成分分析降维的包
from sklearn.cluster import AgglomerativeClustering   #层次聚类

#读取相似度文件
df = pd.read_csv(r'E:\\LearningMaterial\\shujuwajue\\first\\test_sim_7wan.csv',index_col = 0)

'''
传入第一个参数是需要进行层次聚类的数据，这里即可用使用开始读取的数据变量df，第二个参数代表层次聚类选用的方法，第三个参数代表距离计算的方法。
'''
#进行层次聚类
Z = hierarchy.linkage(df, method ='ward',metric='euclidean')

#树状图
hierarchy.dendrogram(Z,labels = df.index,orientation= "left")

#hierarchy.set_link_color_palette(['1', '2', '3', '4'])
#plt.savefig('plot_dendrogram.png')
#plt.rcParams['font.sans-serif'] = ['SimHei']

#plt.title("层次聚类树形图")
plt.show()



#层次聚类结果
#cluster= hierarchy.fcluster(Z, t=1,criterion='inconsistent') #,'inconsistent'
#print ("Original cluster by hierarchy clustering:\n",cluster)


#画散点图
label = cluster.hierarchy.cut_tree(Z,height=0.7)#，我们即可看到在不同的位置裁剪即可得到不同的聚类数目，评价聚类效果
label = label.reshape(label.size,)
#根据两个最大的主成分进行绘图
#print(label)

pca = skldec.PCA(n_components = 0.95)    #选择方差95%的占比
pca.fit(df)   #主城分析时每一行是一个输入数据
result = pca.transform(df)  #计算结果
plt.figure()  #新建一张图进行绘制
plt.scatter(result[:, 0], result[:, 1], c=label, edgecolor='k') #绘制两个主成分组成坐标的散点图
# for i in range(result[:,0].size):
#     plt.text(result[i,0],result[i,1],df.index[i])     #在每个点边上绘制数据名称
x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0]*100.0),2)   #x轴标签字符串
y_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[1]*100.0),2)   #y轴标签字符串
plt.xlabel(x_label)    #绘制x轴标签
plt.ylabel(y_label)    #绘制y轴标签
#plt.title("层次聚类散点图")
plt.show()


