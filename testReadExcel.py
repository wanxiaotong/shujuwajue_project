import time
import pandas as pd
import numpy as np
import jieba
import itertools

###test1
start1=time.time()

latex1 = pd.read_csv(r'mark1.csv', index_col=0) #把标志矩阵读入latex(并把第0列作为索引)

word1=[]
word1 = list(latex1.index) #获取列索引的词汇  存入列表
end1=time.time()

print("time1:")
print(end1-start1)

###test2
start2=time.time()
word2=[]
word3=[]

latex = pd.read_csv(r'mark1.csv', index_col=0, chunksize=200)

for chunk in latex:
    word2 = list(chunk.index) #获取列索引的词汇  存入列表

    word3.extend(word2)
a=chain(word2)
# a=np.concatenate(word2)
# print(word3)
end2=time.time()

print("time2:")
print(end2-start2)
