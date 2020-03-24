import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy  #用于进行层次聚类，画层次聚类图的工具包
from scipy import cluster
from sklearn import decomposition as skldec
import xlrd
# 画热点图, cmap控制颜色画图风格

df = pd.read_csv("file_sim.csv", index_col=0)

# df.index = title
# df.columns = title

sns.set(font='STSong')  # 解决中文字体显示
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df, linewidths=.5, ax=ax, cmap="vlag")

plt.show();