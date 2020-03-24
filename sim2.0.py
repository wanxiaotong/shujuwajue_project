import pandas as pd
import numpy as np
import time

def cos(a, b):  # 相似度计算函数
    dot_product = np.dot(a, b)  #
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


if __name__ == '__main__':
    # start = time.time()  # 开始计时
    #1.读入标志矩阵
    latex = pd.read_csv(r'mark1.csv', index_col=0) #把标志矩阵读入latex(并把第0列作为索引)
    print("latex:")
    print(latex)

    # print(type(latex))#显示类型是dataframe
    words = list(latex.index) #获取列索引的词汇  存入列表
    print("words:")
    print(words)

    L = len(words)

    latex2 = np.array(latex)
    print("latex2：")
    print(latex2)


    #2.读入词向量表
    f =open(r'100000-small.txt', "r", encoding='UTF-8')
    lines = f.readlines()
    #词向量字典
    vertors={}
    for line in lines:
         #print(1)
        #分离出向量
         value = list(map(float,line.split()[1:]))#转换为浮点型
         #{词}='向量'
         vertors[line.split()[0]] = value

    #print (vertors.keys())
    #print(vertors['vpns'])


    #3.构建词相似度矩阵
    #如果词对在标注矩阵里不为0，计算词对的相似度cosine( map(v(i)) , map(v(j)))

    #初始化词相似度矩阵，初始化为0
    #sim_vetor = pd.DataFrame(np.zeros([L, L]), columns=words, index=words)
    #更改数据储存方式

    sim_vetor = np.zeros([L, L])
    # print("type of sim_vetor")
    # print(type(sim_vetor))

    for i in range(0, L):
         # print(i)
         for j in range(i+1, L):
             #遍历词对
              if latex.iloc[i,j] == True:#词对值在标注矩阵里不为0
                   try:
                       #计算词对相似度
                        sim = cos(vertors[words[i]], vertors[words[j]])
                        #print (sim)
                        #写入词相似度矩阵
                        # sim_vetor.iloc[i, j] = sim
                        sim_vetor[i, j] = sim
                   except:
                        next

    # df = pd.DataFrame(sim_vetor)
    # print("type of df")
    # print(type(df))
    sim_vetor = sim_vetor+sim_vetor.T

    print("sim_vetor:")
    print(sim_vetor)

    # df.to_csv('sim2.0.csv', encoding="utf_8_sig")
    np.savetxt(r'sim2.0.csv',sim_vetor,delimiter=',')

    print("ok!")
    # end = time.time()
    # print("运行时间：")
    # print(end - start)


