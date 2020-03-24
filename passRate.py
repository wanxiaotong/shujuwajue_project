import xlrd
import xlwt
from datetime import date,datetime
import pandas as pd
import numpy as np

def level(rr):
    lev=''
    if rr<=0.2:
        lev='极难'
    elif rr<=0.5:
        lev='难'
    elif rr<=0.7:
        lev='中'
    elif rr<=1:
        lev='简单'
    return  lev


filepath=r'E:\LearningMaterial\shujuwajue\passRate\tjoj题解.xls'

sheet = pd.read_excel(filepath,header=None,skiprows=1 )

col1 = sheet[0] # 获取第1列内容
col2 = sheet[1] # 获取第2列内容
col3 = sheet[2] # 获取第3列内容
col4 = sheet[3] # 获取第4列内容
col5 = sheet[4] # 获取第5列内容
# print(col4[2]/col5[2])
# print(len(col2))

ll=len(col4)

data = {'题目编号':sheet[0],'题目网址':sheet[1],'题目名称':sheet[2],'正确数':sheet[3],'提交数':sheet[4],'通过率':{},'难易等级':{}}
last=pd.DataFrame(data)

rate = []  #通过率
leve = []  #难易等级
for i in range(0,ll):  #ll是行数
    r = 0  #通过率
    if col5[i] == 0:   #分母为0时，即提交数为0时
        rate.append(r)  #加到列表里
        leve.append(' ')  #提交数为0时，难易等级那里空着

    else:
        r = col4[i] / col5[i]
        rate.append(r)
        leve.append(level(r)) #调用level函数，得难易等级，加到列表

# print(rate)
last['通过率']=rate
last['难易等级']=leve
print(last)
# last.to_csv(r'E:\LearningMaterial\shujuwajue\passRate\out.csv', encoding="utf_8_sig",index=False)
print('OK!')
