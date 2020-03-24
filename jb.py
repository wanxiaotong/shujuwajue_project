# coding: utf-8
import jieba
import os
import xlrd
import xlwt
import pandas as pd
import numpy as np
import re
from nltk import word_tokenize
import csv

'''分词'''
# test_sent = (
#    "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
#    "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
#    "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凯特琳了。"

# words = jieba.cut(test_sent)  # 用户自定义词典中出现的词条被正确划分
# print("/".join(words))

'''字典'''
# jieba.add_word("我们不需要知道")
# # jieba.del_word("自定义词")  # 删除词条自定义词
#
# print("=" * 100)
# words = jieba.cut(test_sent)  # 再次进行测试
# print("/".join(words))

'''关键词提取'''
# import jieba.analyse
#
# file_name = "E:\LearningMaterial\shujuwajue\BookIntroTen\hebing.txt"
# content = open(file_name, encoding='utf-8').read()
# tags = jieba.analyse.extract_tags(content, topK=10)
# print("/".join(tags))


# import jieba.analyse
# #
# file_name = "E:\LearningMaterial\shujuwajue\BookIntroTen\hebing.txt"
# content = open(file_name, encoding='utf-8').read()
# tags = jieba.analyse.extract_tags(content, topK=10)
# print("/".join(tags))


# from jieba import analyse
#
# textrank = analyse.textrank
# text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
#         是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
#         线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。"
# print("\nkeywords by textrank:")
# # 基于TextRank算法进行关键词抽取
# keywords = textrank(text)
# # 输出抽取出的关键词
# for keyword in keywords:
#     print(keyword + "/",)




'''去掉“博客...”等无用词'''

# stop = ''
#
# file_stop = r'E:\LearningMaterial\shujuwajue\stopwords.txt'  # 停用词表
# # file_text = r'E:\LearningMaterial\shujuwajue\HDUOJ\HDU 1026(3).txt'  # 要处理的文本集合
#
# with open(file_stop, 'r', encoding='utf-8') as f:
#     lines = f.readlines()  # lines是list类型
#     for line in lines:
#         lline = line.strip()  # line 是str类型,strip 去掉\n换行符
#         stop += lline  # 将stop 是列表形式
# print(stop)
#
#
# root = r'E:\LearningMaterial\shujuwajue\HDUOJ'  # 读取的批量txt所在的文件夹的路径
# file_names = os.listdir(root)  # 读取BookIntroTen文件夹下所有的txt的文件名
# file_ob_list = []  # 定义一个列表，用来存放刚才读取的txt文件名
#
# for file_name in file_names:  # 循环地给这10个文件名加上它前面的路径，以得到它的具体路径
#     fileob = root + '\\' + file_name  # 文件夹路径加上\\ 再加上具体要读的的txt的文件名就定位到了这个txt
#     file_ob_list.append(fileob)  # 将路径追加到列表中存储

# for file_ob in file_ob_list:
#     # 读取文本集
#     data = ''
#     after_text = ''
#     with open(file_ob, "r+", encoding='utf-8') as f:
#         data = f.read()
#         # print(data)
#     # with open(file_ob, "w+", encoding='utf-8') as f:
#     #     a = f.read()
#         b = jieba.cut(data)
#         c = ' '.join(b)
#     # print(b)
#         for i in c:
#             if i not in stop:
#                 after_text += i
#         # print(after_text)
#     f.close()
#     with open(file_ob, 'w+') as f:
#         for i in after_text:
#             f.write(i)
#     f.close()

'''分词去停用词去重，无函数版，可行'''
# file_stop = r'E:\LearningMaterial\shujuwajue\TestDocSim\my中文和符号1960.txt'  # 停用词表
# stop = ''
# with open(file_stop, 'r', encoding='utf-8') as f:
#     lines = f.readlines()  # lines是list类型
#     for line in lines:
#         lline = line.replace('\n', ' ')  # line 是str类型,strip 去掉\n换行符
#         stop += lline  # 将stop 是列表形式
# # print(stop)
#
# IO = r'E:\LearningMaterial\shujuwajue\fifty.xlsx'
# sheet = pd.read_excel(io=IO,usecols=0, header=None, names=['tijie'])
# # print(sheet)
#
# words = ''
# text = ''
# for w in sheet['tijie']:
#     w = jieba.cut(w)
#     for word in w:
#         if word not in stop:
#             if word not in text:
#                 text += word
#                 text += ' '
# print(text)
'''保存处理过的数据'''
# f = open(r"E:\LearningMaterial\shujuwajue\cut&quting.txt", "w+")
# f.write(text)
# f.close()

# f = open(r"E:\LearningMaterial\shujuwajue\fiftycut.txt", "w+")
# f.write(words)
# f.close()


'''函数版'''
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

# 相似度计算函数
def cos(a, b):
    dot_product = np.dot(a, b)  #向量点积或矩阵乘法，a1b1+a2b2+...
    norm_a = np.linalg.norm(a)  #np.linalg.norm：求：根号（x1^2+x2^2+...）
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)   #类似于空间两个向量的夹角公式，cos<a,b>

# 计算两个题解的相似度
def get_tijie_sim(set1, set2, sim_vetor):     #sim_vetor：词相似矩阵
    it1 = set1
    it2 = set2
    len1 = len(set1)  # 题解1单词的个数
    len2 = len(set2)  # 题解2单词的个数
    # 初始化文本词矩阵
    sims = pd.DataFrame(np.zeros([len1, len2]))
    m = 0
    # 生成文本词矩阵
    for i in it1:

        n = 0
        for j in it2:
            if i == j:
                sims.loc[m, n] = 1      #loc是根据dataframe的具体标签选取列，而iloc是根据标签所在的位置，从0开始计数。
            else:
                try:
                    sims.loc[m, n] = sim_vetor.loc[i, j]    #有可能报错的代码
                except:
                    sims.loc[m, n] = 0                      #报错后执行的代码
            n = n + 1
        m = m + 1

    # print(sims)

    row = sum(sims.max(axis=0))  # 行方向：单词向量的余弦矩阵，最值和
    col = sum(sims.max(axis=1))  # 列方向：单词向量的余弦矩阵，最值和
    # 把相似度全部为0的行与列统计
    row1 = list(sims.max(axis=0))
    col1 = list(sims.max(axis=1))
    row_zero = row1.count(0)     #统计0出现的次数
    col_zero = col1.count(0)
    # 计算平均相似度
    if(len1 + len2 - row_zero - col_zero == 0):
        sim = 0
    else:
        sim = (row + col) / (len1 + len2 - row_zero - col_zero)

    return sim



szStopWordFile = r'E:\LearningMaterial\shujuwajue\TestDocSim\my中文和符号1960.txt'
#encoding = "unicode_escape"
encoding = 'UTF-8'
strStop = readTxtFile(szStopWordFile, encoding)   #strStop:停用词表所有词，‘ ’类型
# print(strStop)
setStop = buildStopWordList(strStop)              #setStop:停用词表，列表类型
# print(setStop)

IO = r'E:\LearningMaterial\shujuwajue\first\fifty.xlsx'
sheet = pd.read_excel(io=IO,usecols=0, header=None, names=['tijie'])
# print(sheet)

### test1 把每个题解放进一个set()
# text = set()
# for w in sheet['tijie']:
#     w = jieba.cut(w)
#     for word in w:
#         if word not in setStop:
#             text.add(word)

### test1 改进版 设多一个合并set()，调用函数buildWordSet
# single_set = set()
# hebing_set = set()
# for w in sheet['tijie']:
#     single_set = buildWordSet(w, setStop)
#     hebing_set = hebing_set | single_set    # 集合的并集

# print(hebing_set)
# print(len(hebing_set))   #895

### test2 把各个题解放到各个set()，再把每个set()放到列表[]
# setlist = []
# single_set = set()
# lenn = 0
# for w in sheet['tijie']:
#     single_set = buildWordSet(w, setStop)  # 此时single_set是已去停用词的集合
#     # lenn += len(single_set)
#     setlist.append(single_set)  # 将每一个文件的set添加到列表中

### 再改进版(上两个加在一起)(仿照豆瓣)
setlist = []
single_set = set()
for w in sheet['tijie']:
    single_set = buildWordSet(w, setStop)
    setlist.append(single_set)

hebing_set = set()
for i in range(0, len(setlist)):
    hebing_set = hebing_set | setlist[i]  # (通过遍历列表，对每一个文件的set取并集)建立合并文件的set
# print(len(hebing_set))
# print(len(setlist))


# print(setlist)
# print(len(setlist))     # 50
# print(lenn)             #1520
# for i in range(0,len(setlist)):
#     print(len(setlist[i]))
#     lenn += len(setlist[i])
# print(lenn)             #1520

'''形成词矩阵 标志矩阵'''
### test1
# m = len(text)
# print(m)                #895
# data = pd.DataFrame(np.zeros([m, m]), columns=text, index=text)
#
# L = sheet.shape[0]      #shape[0]即行数
# print(L)                #50

### test2
m = len(hebing_set)
# print(m)
data = pd.DataFrame(np.zeros([m, m]), columns=hebing_set, index=hebing_set) #建立词矩阵的表
# print(data)
L = sheet.shape[0]      #shape[0]即行数
# print(L)                #50

'''test1'''
# def removeStopWords(a,setStop):
#     b = set()
#     for word in a:
#         if word not in setStop:
#             b.add(word)
#     return b

## 题解对
# for i in range(0, L - 1):
#     print(i)
#     set1 = jieba.cut(sheet['tijie'][i])  # 题解1
#     # print(type(set1))      #set1类型：generator 生成器 。在Python中，这种一边循环一边计算的机制，称为生成器（Generator）
#     set3 = removeStopWords(set1, setStop)
#     # print(set3)
#     for j in range(i + 1, L):
#
#         set2 = jieba.cut(sheet['tijie'][j])  # 题解2
#         set4 = removeStopWords(set2, setStop)
#         v = list(set4)
#
#         # 题解对里的词
#         for w1 in set3:
#             # 将要计算相似度的词汇矩阵+1
#             data.loc[w1][v] = data.loc[w1][v] + 1
#             # 同一个词不用计算相似度，置为0
#             data.loc[w1, w1] = 0
#             # data.loc[v][w1]=data.loc[v][w1]+1
#
# data = data + data.values.T
# # 将标注矩阵写入Excel
# print('ok!')

# data.to_csv(r'E:\LearningMaterial\shujuwajue\test1.csv', encoding='utf_8_sig')

# data.to_csv('test1.csv')

'''test2'''
# for i in range(0, L):
#     print(i)
#     set1 = setlist[i]              #AllSet_list是一个存储所有文件的set的列表
#
#     for j in range(i+1, L):
#         set2 = setlist[j]
#         v = list(set2)   #把set 转化成 list
#
#         #文件对里的词
#         for w1 in set1:
#             #将要计算相似度的词汇矩阵+1
#             data.loc[w1][v]=data.loc[w1][v]+1  #可以用列表作为参数，一次性设置
#             #同一个词不用计算相似度，置为0
#             data.loc[w1, w1] = 0
#             # data.loc[v][w1]=data.loc[v][w1]+1
#
# data = data + data.values.T #矩阵是对称的
# #将标注矩阵写入Excel
# print("ok!")
#
# data.to_csv(r'E:\LearningMaterial\shujuwajue\test2.csv', encoding="utf_8_sig")

################################## 在外置0
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

data = data | data.values.T  # 矩阵是对称的




'''相似矩阵'''
### 1.读入标志矩阵
latex = pd.read_csv(r'E:\LearningMaterial\shujuwajue\first\test2.csv', index_col = 0) # 把标志矩阵读入latex(并把第0列作为行索引)
#print(latex)
words = list(latex.index) # 获取列索引的词汇  存入列表
LL= len(words)

### 2.读入词向量表
f =open(r'E:\LearningMaterial\shujuwajue\TestDocSim\非正式腾讯语料库子集\70000-small.txt', "r", encoding='UTF-8')
lines = f.readlines()
# 词向量字典
vertors={}
for line in lines:
     # print(1)
    # 分离出向量
     value = list(map(float, line.split()[1:]))  # 将line转换为浮点型。map(function, parameter): 会根据提供的函数对指定序列做映射
     # {词=['向量'组（列表）]}
     vertors[line.split()[0]] = value    # 语料库里第0列是词，[1:]是数字。 vertors字典的键是文字，值是一个列表

# print (vertors.keys())  #字典的键
# print(vertors['右下方'])

### 3.构建词相似度矩阵
# 如果词对在标注矩阵里不为0，计算词对的相似度cosine( map(v(i)) , map(v(j)))
# 初始化词相似度矩阵，初始化为0

sim_vetor = pd.DataFrame(np.zeros([LL, LL]), columns=words, index=words)

for i in range(0, LL):    # 0 - 895
    print(i)
    for j in range(i+1, LL):
         # 遍历词对
        if latex.iloc[i, j] != 0: # 词对值在标注矩阵里不为0。 iloc是根据标签所在的位置，从0开始计数。
            try:
                # 计算词对相似度
                sim = cos(vertors[words[i]],vertors[words[j]])
                #print (sim)
                #写入词相似度矩阵
                sim_vetor.iloc[i, j] = sim
            except:
                next

sim_vetor = sim_vetor + sim_vetor.values.T
print ('ok!')
print (sim_vetor)
# sim_vetor.to_csv(r'E:\LearningMaterial\shujuwajue\test_sim_50wan.csv', encoding="utf_8_sig")


######### 题解之间的相似度比较
file_sim = pd.DataFrame(np.zeros([L, L]))  # 题解之间的相似矩阵  L:50 (题解个数)

sim_vector = pd.read_csv(r'E:\LearningMaterial\shujuwajue\first\test_sim_50wan.csv', index_col=0)  # 读入词相似矩阵excel
# print(sim_csv)

for i in range(0, L):
    print(i)
    for j in range(i + 1, L):
        file_sim.iloc[i, j] = get_tijie_sim(setlist[i], setlist[j], sim_vector)

print(file_sim)
# file_sim.to_csv(r'E:\LearningMaterial\shujuwajue\first\test_file2.csv', encoding="utf_8_sig",na_rep='NA')
