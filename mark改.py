"""
这份代码修改了if __name__ == '__main__'这一部分中两层迭代的部分。
原始代码中，__main__里面两次外层迭代的之间的间隔：3'14

代码修改建议：
1. DataFrame的at方法虽然一次只能存取一个数值，但是整体时间还是比使用loc要快。（两次外层迭代时间缩短到1'50）
2. 在标注矩阵对角线上置零的工作可以提到循环之外进行。
3. 标注矩阵可以使用bool类型，降低内存使用量并提升速度。（0'58）
"""
import time
import jieba
import pandas as pd
import numpy as np
import datetime


def readTxtFile(filename, ec='gb2313') :
    # 系统默认gb2312， 大文件常用'UTF-8'
    str = ""
    with open(filename, "r", encoding=ec) as f :  # 设置文件对象
        str = f.read()  # 可以是随便对文件的操作
    #     regex = re.compile(r'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
    #                                              #过滤掉非中文字符
    #     regex2 = regex.sub("", str)
    # return(regex2)
    return str


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


def buildStopWordList(strStop) :
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

# def buildSimMatrix(set1,set2):
#     for w1 in set1:
#         for w2 in set2:
#             data.at[w1, w2] = True  # 使用at方法整体上快于loc方法


if __name__ == '__main__' :
    ##########标志矩阵
    start1 = time.time()  # 开始计时

    szStopWordFile = r'my中文和符号1960.txt'

    encoding = 'UTF-8'
    strStop = readTxtFile(szStopWordFile, encoding)  # strStop:停用词表所有词，‘ ’类型
    # print(strStop)
    setStop = buildStopWordList(strStop)  # setStop:停用词表，列表类型
    # print(setStop)

    IO = r'fifty.xlsx'
    sheet = pd.read_excel(io=IO, usecols=0, header=None, names=['tijie'])
    # print(sheet)

    end1=time.time()
    time1=end1-start1
    print('1-ok!  ,time1:(读取停用词表和题解原数据)')
    print(time1)


    start2 = time.time()  # 开始计时

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

    end2 = time.time()
    time2 = end2 - start2
    print('2-ok!  ,time2:（分词）' )
    print(time2)


    start3 = time.time()  # 开始计时

    for i in range(0, L):
        # print(i, datetime.datetime.now())
        set1 = setlist[i]  # type(set1):'set'

        for j in range(i + 1, L):
            set2 = setlist[j]
            # buildSimMatrix(set1,set2)

            # 文件对里的词
            for w1 in set1:
                # 将要计算相似度的词汇矩阵 + 1
                for w2 in set2:
                    data.at[w1, w2] = True  # 使用at方法整体上快于loc方法

            # # 文件对里的词
            # for w1,w2 in zip(set1,set2):
            #     # 将要计算相似度的词汇矩阵 + 1
            #     data.at[w1, w2] = True  # 使用at方法整体上快于loc方法

    # print(type(set1))
    # print(len(set2))

    # print("3-ok!")

    # 同一个词不用计算相似度，置零
    for i in range(m):
        data.iat[i, i] = False

    data = data | data.values.T  # 矩阵是对称的

    end3 = time.time()
    time3 = end3 - start3
    print('3-ok!  ,time3:（形成标志矩阵，4个for循环）' )
    print(time3)

    # 将标注矩阵写入Excel
    # print(data)
    # print("4-ok!")

    # data.to_csv(r'markAll.csv', encoding="utf_8_sig")