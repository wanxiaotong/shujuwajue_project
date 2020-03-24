import pandas as pd
import numpy as np
import time

def cos(a, b):  # 相似度计算函数
    dot_product = np.dot(a, b)   #向量点积或矩阵乘法，a1b1+a2b2+...
    norm_a = np.linalg.norm(a)   #np.linalg.norm：求：根号（x1^2+x2^2+...）
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)   #类似于空间两个向量的夹角公式，cos<a,b>

if __name__ == '__main__':
    #1.读入标志矩阵
    latex = pd.read_csv(r'mark1.csv', index_col=0) #把标志矩阵读入latex(并把第0列作为索引)
    # print(type(latex))
    words = list(latex.index) #获取列索引的词汇  存入列表
    # print(type(words))
    L = len(words)

    print('1-ok!')

    #2.读入词向量表


    f =open(r'100000-small.txt', "r", encoding='UTF-8')
    lines = f.readlines()
    #词向量字典
    vertors={}
    for line in lines:
         # print(1)
        #分离出向量
         value = list(map(float,line.split()[1:]))#转换为浮点型
         #{词=['向量'组（列表）]}
         vertors[line.split()[0]] = value

    # print (vertors)
    #print(vertors['vpns'])

    print('2-ok!')


    #3.构建词相似度矩阵
    #如果词对在标注矩阵里不为0，计算词对的相似度cosine( map(v(i)) , map(v(j)))

    #初始化词相似度矩阵，初始化为0

    start=time.time()

    sim_vetor = pd.DataFrame(np.zeros([L, L]), columns=words, index=words)

    for i in range(0, L):
         for j in range(i+1, L):
             #遍历词对
              if latex.iloc[i,j]!=0:#词对值在标注矩阵里不为0
                   try:
                       #计算词对相似度
                        sim = cos(vertors[words[i]], vertors[words[j]])
                        #print (sim)
                        #写入词相似度矩阵
                        sim_vetor.iloc[i, j] = sim
                   except:
                        next

    sim_vetor = sim_vetor+sim_vetor.values.T

    print(sim_vetor)

    end = time.time()
    print('time:')
    print(end - start)

    print('3-ok!')




    # sim_vetor.to_csv('simSet.csv', encoding="utf_8_sig")

