import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from gensim.models import KeyedVectors


def readTxtFile(filename, ec='utf-8'): # 系统默认gb2312， 大文件常用'UTF-8'
    str=""
    with open(filename, "r", encoding=ec) as f:  # 设置文件对象
        str = f.read()  # 可以是随便对文件的操作
    #     regex = re.compile(r'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
    #                                              #过滤掉非中文字符
    #     regex2 = regex.sub("", str)
    # return(regex2)
    return (str)

def buildWordSet(str, setStop):  #根据停用词过滤，并利用set去重
    # 将分词、去停用词后的文本数据存储在list类型的texts中
    words = ' '.join(jieba.cut(str)).split(' ')  # 利用jieba工具进行中文分词
    setStr = set()
    # 过滤停用词，只保留不属于停用词的词语
    for word in words:
        if word not in setStop:
            setStr.add(word)
    # if (len(word)==0)
    #     word =
    return setStr

def buildStopWordList(strStop):
    stopwords = set()
    strSplit = strStop.split('\n')
    # print(strSplit)
    for line in strSplit:
        # print(line)
        stopwords.add(line.strip())
    stopwords.add('\n')
    stopwords.add('\t')
    stopwords.add(' ')
    stopwords.add('')
    return stopwords


def get_file_sim(set1, set2, wv):
    it1 = set1
    it2 = set2
    len1 = len(set1)  # 文本1单词的个数
    len2 = len(set2)  # 文本2单词的个数
    # print("len1:", len1)
    # print("len2:", len2)
    # 初始化文本词矩阵
    sims = pd.DataFrame(np.zeros([len1, len2]))
    m = 0
    # 生成文本词矩阵
    for i in it1:

        n = 0
        for j in it2:
            if i == j:
                sims.loc[m, n] = 1
            else:
                try:
                    sims.loc[m, n] = wv.similarity(i, j)  # 计算两个词相似度
                except:
                    sims.loc[m, n] = 0
            n = n + 1
        m = m + 1

    # print(sims)

    row = sum(sims.max(axis=0))  # 行方向：单词向量的余弦矩阵，最值和
    col = sum(sims.max(axis=1))  # 列方向：单词向量的余弦矩阵，最值和

    # print("row:", row)
    # print("col:", col)
    # 把相似度全部为0的行与列统计
    row1 = list(sims.max(axis=0))

    # print("row1", row1)
    col1 = list(sims.max(axis=1))
    row_zero = row1.count(0)
    col_zero = col1.count(0)
    # print("row_zero:", row_zero)
    # print("col_zero:", col_zero)
    # 计算平均相似度
    if (len1 + len2 - row_zero - col_zero != 0):
        sim = (row + col) / (len1 + len2 - row_zero - col_zero)
    else:
        sim = 0

    return sim


if __name__ == '__main__':

    szStopWordFile = r'my中文和符号1960.txt'

    encoding = 'UTF-8'
    strStop = readTxtFile(szStopWordFile, encoding)  # strStop:停用词表所有词，‘ ’类型
    # print(strStop)
    setStop = buildStopWordList(strStop)  # setStop:停用词表，列表类型
    # print(setStop)

    IO = r'fifty.xlsx'
    sheet = pd.read_excel(io=IO, usecols=0, header=None, names=['tijie'])
    # print(sheet)


    setlist = []
    single_set = set()
    for w in sheet['tijie']:
        single_set = buildWordSet(w, setStop)
        setlist.append(single_set)


    L = sheet.shape[0]  # shape[0]即行数

     #########################文件之间的相似度比较

    ##L = lstFile.__len__()  # 文件总个数
    ##L = sheet.shape[0]      #shape[0]即行数
    start=time.time()

    file_sim = pd.DataFrame(np.zeros([L, L])) #文件之间的相似矩阵

    # sim_vector = pd.read_csv(r'sim.csv', index_col=0) #读入相似矩阵excel

    # 加载语料库
    szTXcorpus = r"100000-small.txt"  # 小语料库
    model = KeyedVectors.load_word2vec_format(szTXcorpus, binary=False)

    for i in range(0, L):
        print(i)
        for j in range(i+1, L):
            file_sim.iloc[i, j] = get_file_sim(setlist[i], setlist[j], model)

###############test#############

    # file_sim = np.zeros([L, L]) # 文件之间的相似矩阵
    # print('1-ok')
    # sim_vector = pd.read_csv(r'sim.csv', index_col=0)  # 读入相似矩阵excel
    #
    # for i in range(0, L):
    #     print(i)
    #     for j in range(i + 1, L):
    #         file_sim[i, j] = get_file_sim(setlist[i], setlist[j], sim_vector)

    ################
    end = time.time()
    print(end - start)

    # print(setlist[0], setlist[1])
    # print("----------------------------")
    # print(get_file_sim(setlist[0], setlist[1], sim_vector));
    print(file_sim)
    #print(setlist[1])

    # file_sim.to_csv('file_sim.csv', encoding="utf_8_sig")

