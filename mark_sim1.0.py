import jieba
import pandas as pd
import numpy as np
import time


def readTxtFile(filename, ec='gb2313') :
    # 系统默认gb2312， 大文件常用'UTF-8'
    str = ""
    with open(filename, "r", encoding=ec) as f :  # 设置文件对象
        str = f.read()  # 可以是随便对文件的操作
    return str

def readExcelFile(filename, ec='gb2313'):##读取Excel文件
    str=""
    with open(filename, "r", encoding=ec) as f:
        sheet = pd.read_excel(io=filename, usecols=0, header=None, names=['tijie'])
        # print("sheet__"+sheet)
    return sheet

def buildWordSet(str, setStop) :  # 根据停用词过滤，并利用set去重
    # 将分词、去停用词后的文本数据存储在list类型的texts中
    words = ' '.join(jieba.cut(str)).split(' ')  # 利用jieba工具进行中文分词
    setStr = set()
    # 过滤停用词，只保留不属于停用词的词语
    for word in words :
        if word not in setStop :
            setStr.add(word)
    # if (len(word)==0)
    #     word =
    return setStr


def buildStopWordList(strStop) :      #getVaDis.py for循环用stopwords = {line.strip() for line in strSplit}
    stopwords = set()
    strSplit = strStop.split('\n')
    # print(strSplit)
    for line in strSplit :
        # print(line)
        stopwords.add(line.strip())
    stopwords.add('\n')
    stopwords.add('\t')
    stopwords.add(' ')
    stopwords.add('')
    return stopwords

def cos(a, b):  # 相似度计算函数
    dot_product = np.dot(a, b)  #
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


if __name__ == '__main__' :
    ##########标志矩阵

    start = time.time()  # 开始计时

    szStopWordFile = r'E:\programming\pythonpt\dataDig\my中文和符号1960.txt'

    encoding = 'UTF-8'
    strStop = readTxtFile(szStopWordFile, encoding)  # strStop:停用词表所有词，‘ ’类型
    # print(strStop)
    setStop = buildStopWordList(strStop)  # setStop:停用词表，列表类型
    # print(setStop)

    IO = r'E:\programming\pythonpt\dataDig\fifty.xlsx'

    sheet = readExcelFile(IO, encoding)
    # print(sheet)




    setlist = []
    single_set = set()
    for w in sheet['tijie'] :
        single_set = buildWordSet(w, setStop)
        setlist.append(single_set)

    hebing_set = set()
    for i in range(0, len(setlist)) :
        hebing_set = hebing_set | setlist[i]  # (通过遍历列表，对每一个文件的set取并集)建立合并文件的set

    m = len(hebing_set)
    # print(m)
    data = pd.DataFrame(np.zeros([m, m], dtype=bool), columns=hebing_set, index=hebing_set)  # 建立词矩阵的表
    # print(data)
    L = sheet.shape[0]  # shape[0]即行数
    # print(L)                #50




    for i in range(0, L):
        set1 = setlist[i]  # AllSet_list是一个存储所有文件的set的列表

        for j in range(i + 1, L):
            set2 = setlist[j]

            # 文件对里的词
            for w1 in set1:
                # 将要计算相似度的词汇矩阵 + 1
                for w2 in set2:
                    data.at[w1, w2] = True  # 使用at方法整体上快于loc方法


    # 同一个词不用计算相似度，置零
    for i in range(m):
        data.iat[i, i] = False

    data = data | data.values.T  # 矩阵是对称的`

    # 将标注矩阵写入Excel
    # print("ok!")
    # print(data)


    data.to_csv(r'mark_sim1.0.csv', encoding="utf_8_sig")


############# sim.py #################
    # 1.读入标志矩阵
    latex = pd.read_csv(r'mark_sim1.0.csv', index_col=0)  # 把标志矩阵读入latex(并把第0列作为索引)
    print("latex:")
    print(latex)

    # print(type(latex))#显示类型是dataframe
    words = list(latex.index)  # 获取列索引的词汇  存入列表
    print("words:")
    print(words)

    L = len(words)

    latex2 = np.array(latex)
    print("latex2：")
    print(latex2)

    # 2.读入词向量表
    f = open(r'E:\programming\pythonpt\dataDig\100000-small.txt', "r", encoding='UTF-8')
    lines = f.readlines()
    # 词向量字典
    vertors = {}
    for line in lines:
        # print(1)
        # 分离出向量
        value = list(map(float, line.split()[1:]))  # 转换为浮点型
        # {词}='向量'
        vertors[line.split()[0]] = value

    # print (vertors.keys())
    # print(vertors['vpns'])

    # 3.构建词相似度矩阵
    # 如果词对在标注矩阵里不为0，计算词对的相似度cosine( map(v(i)) , map(v(j)))

    # 初始化词相似度矩阵，初始化为0
    # sim_vetor = pd.DataFrame(np.zeros([L, L]), columns=words, index=words)
    # 更改数据储存方式

    sim_vetor = np.zeros([L, L])
    # print("type of sim_vetor")
    # print(type(sim_vetor))

    for i in range(0, L):
        # print(i)
        for j in range(i + 1, L):
            # 遍历词对
            if latex.iloc[i, j] == True:  # 词对值在标注矩阵里不为0
                try:
                    # 计算词对相似度
                    sim = cos(vertors[words[i]], vertors[words[j]])
                    # print (sim)
                    # 写入词相似度矩阵
                    # sim_vetor.iloc[i, j] = sim
                    sim_vetor[i, j] = sim
                except:
                    next

    # df = pd.DataFrame(sim_vetor)
    # print("type of df")
    # print(type(df))
    sim_vetor = sim_vetor + sim_vetor.T

    print("sim_vetor:")
    print(sim_vetor)
    np.savetxt(r'sim2.0.csv',sim_vetor,delimiter=',')

    print("ok!")

    end = time.time()
    print("运行时间：")
    print(end - start)


