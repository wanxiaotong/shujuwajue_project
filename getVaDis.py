'''
v1,未改进版本用时：81.7  81.5s
v2，改进点：使用字典记录已经计算过的词语义 79.4 76.39秒
v3,有1的行和列不再计算，用时47.4秒  40.8秒
v4,使用numpy矩阵，用时31.3秒    35.6秒
'''

import jieba
import os
import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
import matplotlib.pyplot as plt
import seaborn as sns
import docx

class getVaDis:
    record=dict()   #稀疏矩阵，用于记录已经计算过了的词相似度
    #读取txt文档
    def readTxtFile(self,filename, ec='gb2312'):  # 系统默认gb2312
        str = ""
        with open(filename, "r", encoding=ec) as f:  # 设置文件对象
            str = f.read()  # 可以是随便对文件的操作
        return (str)
    #读取doc文档
    def readDocFile(self,filename):  # 系统默认gb2312
        str = ""
        file = docx.Document(filename)
        for p in file.paragraphs:
            str+=p.text
        return str

    def buildStopWordList(self,strStop):  # 传入停用词字符串，获得停用词集合
        strSplit = strStop.split('\n')  # 优化点1
        stopwords = {line.strip() for line in strSplit}
        stopwords.add('\n')
        stopwords.add('\t')
        stopwords.add(' ')
        return stopwords

    def buildWordSet(self,str, setStop):  # 根据停用词setStop集合过滤，并利用set去重
        # 将分词、去停用词后的文本数据存储在list类型的texts中
        words = set(jieba.cut(str))  # 利用jieba工具进行中文分词
        # 过滤停用词，只保留不属于停用词的词语，使用集合推导式
        setStr = {word for word in words if word not in setStop}
        return setStr


    def readDir(self,szDir):  # 读取目录
        lstFile = []
        for file in os.listdir(szDir):
            file_path = os.path.join(szDir, file)
            lstFile.append(file_path)
        return lstFile


    ################################### GenSim相关语义计算
    def loadTXModel(self,szTXcorpus):
        wv_from_text = KeyedVectors.load_word2vec_format(szTXcorpus, binary=False)
        return wv_from_text


    def compSim(self,set1, set2, wv):
        N1 = len(set1)
        N2 = len(set2)
        fSim = 0
        df = np.zeros([N1, N2])
        i = 0
        j = 0
        recordCol=set()
        for w1 in set1:
            j = 0
            for w2 in set2:
                if(j in recordCol):
                    j+=1
                    continue
                if (w1 == w2):
                    df[i, j] = 1.0
                    recordCol.add(j)
                    break
                if (((w1, w2) not in self.record.keys()) and ((w2, w1) not in self.record.keys())):
                    try:
                        f = wv.similarity(w1, w2)  # 计算两个词相似度
                        # print('1', (w1, w2))
                        self.record[(w1, w2)] = f
                    except:
                        # print('2', (w1, w2))
                        f = 0
                        self.record[(w1, w2)] = f
                    df[i, j] = f
                else:
                    try:
                        # print('3', (w1, w2))
                        df[i, j] = self.record[(w1, w2)]
                    except:
                        # print('4', (w1, w2))
                        df[i, j] = self.record[(w2, w1)]
                j = j + 1
            i = i + 1
            # print(self.record)
        # (1) 行列都取最大值，然后合起来求平均  -------- 效果最好
        v0 = df.max(axis=0)
        v1 = df.max(axis=1)
        fSim = (v0.sum() + v1.sum()) / (len(v0) + len(v1))
        print(df)
        print(v0)
        print(v1)
        return fSim


if __name__ == "__main__":
    start = time.time()  # 开始计时
    test=getVaDis()

    #################  读取停用词列表
    szStopWordFile = r'./my中文和符号1960.txt'
    encoding = 'UTF-8'
    strStop = test.readTxtFile(szStopWordFile,encoding)
    #print(strStop)
    setStop = test.buildStopWordList(strStop)  # 生成停用词集合
    #print(setStop)

    #################  读取一个文件夹中的的各个比较文档
    szDir = r'fifty.xlsx'
    lstFile = test.readDir(szDir)
    print(lstFile)

    # 获取短文件名列表
    lstFileShow = []
    for szFilename in lstFile:
        name=szFilename.split('\\')[1]
        lstFileShow.append(re.findall('(.+).docx\Z',name)[0])
    print(lstFileShow)

    # 读取各个文档并分词、去停用词到 一个list. lstSetDocContent[i] 为 第i-1个，内容为一个文档对应的set
    lstSetDocContent = []
    for szFileName in lstFile:
        str = test.readDocFile(szFileName)
        setStr = test.buildWordSet(str, setStop)
        lstSetDocContent.append(setStr)
    # 加载语料库
    szTXcorpus = r"100000-small.txt"  # 小语料库
    wv = test.loadTXModel(szTXcorpus)
    #创建一个文本语义相似度矩阵
    L = len(lstSetDocContent)  # L个文档
    df1 = pd.DataFrame(np.zeros([L, L]),index=lstFileShow,columns=lstFileShow)
    ############  双层循环，低效，需要改进

    for i in range(0, L - 1):  # from 0 to L-2
        set1 = lstSetDocContent[i]
        for j in range(i + 1, L):  # from i+1 to L
            set2 = lstSetDocContent[j]
            df1.iloc[i, j] = test.compSim(set1, set2, wv)  # 计算两个set组的相似度

    end = time.time()
    df1 = df1 + df1.values.T  # 加上转置
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    print(df1)


    #计算时间
    print(end - start)
    # #画热力图
    # df1.index = lstFileShow
    # df1.columns = lstFileShow
    # sns.set(font='simhei')  # 解决中文字体显示
    # f, ax = plt.subplots(figsize=(9, 6))
    # sns.heatmap(df1, annot=False, linewidths=.5, ax=ax, cmap="vlag")
    # plt.show()
