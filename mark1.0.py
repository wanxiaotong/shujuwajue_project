import jieba
import pandas as pd
import numpy as np

def readTxtFile(filename, ec='gb2313'):
    # 系统默认gb2312， 大文件常用'UTF-8'
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

if __name__ == '__main__':
    ##########标志矩阵

    szStopWordFile = r'my中文和符号1960.txt'

    encoding = 'UTF-8'
    strStop = readTxtFile(szStopWordFile, encoding)   #strStop:停用词表所有词，‘ ’类型
    # print(strStop)
    setStop = buildStopWordList(strStop)              #setStop:停用词表，列表类型
    # print(setStop)

    IO = r'demo1zoj.xlsx'
    sheet = pd.read_excel(io=IO, usecols=0, header=None, names=['tijie'])
    # print(sheet)

    setlist = []
    single_set = set()
    for w in sheet['tijie']:
        single_set = buildWordSet(w, setStop)
        setlist.append(single_set)

    hebing_set = set()
    for i in range(0, len(setlist)):
        hebing_set = hebing_set | setlist[i]  # (通过遍历列表，对每一个文件的set取并集)建立合并文件的set

    m = len(hebing_set)
    # print(m)
    data = pd.DataFrame(np.zeros([m, m]), columns=hebing_set, index=hebing_set) #建立词矩阵的表
    # print(data)
    L = sheet.shape[0]      #shape[0]即行数
    # print(L)                #50

    for i in range(0, L):
        print(i)
        set1 = setlist[i]              #AllSet_list是一个存储所有文件的set的列表

        for j in range(i+1, L):
            set2 = setlist[j]
            v = list(set2)   #把set 转化成 list

            #文件对里的词
            for w1 in set1:
                #将要计算相似度的词汇矩阵+1
                data.loc[w1][v]=data.loc[w1][v]+1  #可以用列表作为参数，一次性设置
                #同一个词不用计算相似度，置为0
                data.loc[w1, w1] = 0
                # data.loc[v][w1]=data.loc[v][w1]+1

    data = data + data.values.T #矩阵是对称的
    #将标注矩阵写入Excel
    print("ok!")

    data.to_csv(r'mark.csv', encoding="utf_8_sig")