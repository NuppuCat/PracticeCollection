# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 08:19:14 2021

@author: One
"""

import numpy as np

def Forward(trainsition_probability,emission_probability,pi,obs_seq):
    """
    :param trainsition_probability:trainsition_probability是状态转移矩阵
    :param emission_probability: emission_probability是发射矩阵
    :param pi: pi是初始状态概率
    :param obs_seq: obs_seq是观察状态序列
    :return: 返回结果
    """
    trainsition_probability = np.array(trainsition_probability)
    emission_probability  = np.array(emission_probability)
    # print(emission_probability[:,0])
    pi = np.array(pi)
    Row = np.array(trainsition_probability).shape[0]
    Col = len(obs_seq)
    F = np.zeros((Row,Col))                      #最后要返回的就是F，就是我们公式中的alpha
    F[:,0] = pi * np.transpose(emission_probability[:,obs_seq[0]])  #这是初始化求第一列,就是初始的概率*各自的发射概率
    # print( F[:,0])
    for t in range(1,len(obs_seq)):              #这里相当于填矩阵的元素值
        for n in range(Row):                     #n是代表隐藏状态的
            F[n,t] = np.dot(F[:,t-1],trainsition_probability[:,n])*emission_probability[n,obs_seq[t]]   #对应于公式,前面是对应相乘
            # print(F[n,t])

    return F

def Backward(trainsition_probability,emission_probability,pi,obs_seq):
    """
    :param trainsition_probability:trainsition_probability是状态转移矩阵
    :param emission_probability: emission_probability是发射矩阵
    :param pi: pi是初始状态概率
    :param obs_seq: obs_seq是观察状态序列
    :return: 返回结果
    """
    trainsition_probability = np.array(trainsition_probability)
    emission_probability = np.array(emission_probability)
    pi = np.array(pi)                 #要进行矩阵运算，先变为array类型

    Row = trainsition_probability.shape[0]
    Col = len(obs_seq)
    F = np.zeros((Row,Col))
    F[:,(Col-1):] = 1                  #最后的每一个元素赋值为1

    for t in reversed(range(Col-1)):
        for n in range(Row):
            F[n,t] = np.sum(F[:,t+1]*trainsition_probability[n,:]*emission_probability[:,obs_seq[t+1]])


    return F

if __name__ == '__main__':
    # emission_probability= [[ 0.6, 0.4, 0, 0 , 0 , 0 , 0, 0, 0, 0, 0, 0, 0, 0],
    #                            [0, 0, 0.2, 0.4 , 0.4 , 0 , 0, 0, 0, 0, 0, 0, 0, 0],
    #                            [0, 0, 0, 0 , 0 , 0.5 , 0.3, 0.2, 0, 0, 0, 0, 0, 0],
    #                            [ 0, 0, 0, 0 , 0 , 0 , 0, 0, 0.1, 0.4 , 0.5, 0, 0, 0],
    #                            [0, 0, 0, 0 , 0 , 0 , 0, 0, 0, 0, 0, 0.3, 0.4 , 0.3]]
    # trainsition_probability= [[0 , 0, 0.5, 0 , 0.5],
    #                         [1, 0, 0, 0 , 0],
    #                         [0 , 0, 0.3, 0 , 0.7],
    #                         [0 , 1, 0, 0 , 0],
    #                         [0 , 0.5, 0, 0.5 , 0]]
    emission_probability = [[0.25,0.15,0.15,0,0.2,0.25],[0,0.3,0.5,0.2,0,0],[0.15,0.15,0.2,0.3,0,0.2]]
    trainsition_probability = [[0.2,0.3,0.5],[0.4,0.1,0.5],[0.3,0.7,0]]
    pi = [0.3,0.3,0.4]
    trainsition_probability = np.array(trainsition_probability)
    emission_probability  = np.array(emission_probability)
    # print(emission_probability[:,0])
    pi = np.array(pi)

    #然后下面先得到前向算法,在A,B,pi参数已知的前提下，求出特定观察序列的概率是多少?
    obs_seq = [4,5,2,1,3]
    Row = np.array(trainsition_probability).shape[0]
    Col = len(obs_seq)

    F_forward = Forward(trainsition_probability,emission_probability,pi,obs_seq)                  #得到前向算法的结果
    F_backward = Backward(trainsition_probability,emission_probability,pi,obs_seq)        #得到后向算法的结果

    res_forward = 0
    for i in range(Row):                         #将最后一列相加就得到了我们最终的结果
        res_forward+=F_forward[i][Col-1]                          #求和于最后一列
    emission_probability = np.array(emission_probability)
    
    #下面是得到后向算法的结果
    res_backword = 0
    res_backward = np.sum(pi*F_backward[:,0]*emission_probability[:,obs_seq[0]])           #一定要乘以发射的那一列
    # print ("res_backward = {}".format(res_backward))
    # print ("res_forward = {}".format(res_forward))
    a = np.sum(np.sum(np.multiply(F_forward,F_backward)))
    print(a)
    
#%%  7.2
import numpy as np
#第一个参数表示转移概率矩阵 $ P(Z_i|Z_{i-1})$,第二个参数表示发射概率$P(W_i|Z_i)$,第三个参数表示隐藏变量Z的初始概率 $P(Z_i)$,第四个参数表示要标注的句子对像。
def viterbi(trainsition_probability,emission_probability,pi,obs_seq):
    #转换为矩阵进行运算
    trainsition_probability=np.array(trainsition_probability)
    emission_probability=np.array(emission_probability)
    pi=np.array(pi)
     #句子中的词在词库中的下标位置
    # 最后返回一个Row*Col的矩阵结果
    Row = np.array(trainsition_probability).shape[0]#获取词性的个数
    Col = len(obs_seq)#获取句子中词的个数
    #定义要返回的矩阵，即动态规划中需要维护的矩阵，计算矩阵中的每一个元素值
    F=np.zeros((Row,Col))#行数代表词的个数，列数代表词性的种类
    
    #初始状态
    F[:,0]=pi*np.transpose(emission_probability[:,obs_seq[0]])
    for t in range(1,Col):#针对每一列
        list_max=[]
        for n in range(Row):#遍历每一行
            list_x=list(np.array(F[:,t-1])*np.transpose(trainsition_probability[:,n]))
            #获取最大概率
            list_p=[]
            for i in list_x:
                list_p.append(i*10000)
            list_max.append(max(list_p)/10000)
        F[:,t]=np.array(list_max)*np.transpose(emission_probability[:,obs_seq[t]])
    return F

if __name__=='__main__':
    #隐藏状态
    invisible=['Sunny','Cloud','Rainy']
    #初始状态
    pi=[0.2,0.2,0.2,0.2,0.2]
    #转移矩阵
    trainsion_probility=[[0 , 0 , 0.3, 0.3, 0.4],
                         [0 , 0 , 0, 0.4, 0.6],
                         [0 , 0.75 , 0.25 , 0, 0],
                         [0 , 0 , 0, 0, 1],
                         [0 , 0 , 1, 0, 0]]
    #发射矩阵
    emission_probility=[[0.4 , 0.3, 0.2, 0.1, 0, 0, 0 , 0 , 0 , 0, 0, 0 , 0 , 0 , 0, 0],
                        [0.3, 0.4 , 0 , 0.3, 0, 0, 0 , 0 , 0 , 0, 0, 0 , 0 , 0 , 0, 0],
                        [0, 0, 0 , 0 , 0.1 , 0.15 , 0.15, 0.15, 0.15, 0.1 , 0.1, 0.1, 0 , 0 , 0, 0],
                        [0, 0, 0 , 0 , 0, 0, 0 , 0 , 0 , 0, 0.3, 0 , 0 , 0 , 0.4 , 0.3],
                        [0, 0, 0 , 0.05, 0.05 , 0.05 , 0.1, 0 , 0 , 0.15 , 0.1, 0.15, 0.2, 0.15, 0, 0]]
    #最后显示状态
    obs_seq=[3,11,3,4,9,0,10,13]
    #最后返回一个Row*Col的矩阵结果
    F=viterbi(trainsion_probility,emission_probility,pi,obs_seq)
    
    print(np.argmax(F,axis=0))
#%%  7.3
import numpy
    #前向-后向算法(Baum-Welch算法):由 EM算法 & HMM 结合形成
def baum_welch(A,B,Pi,O,e=0.05):
    #A trans b Emi
    row=A.shape[0]
    col=len(O)

    done=False
    while not done:
        zeta=numpy.zeros((row,row,col-1))
        alpha=Forward(A,B,Pi,O)
        beta=Backward(A,B,Pi,O)
        #EM算法：由 E-步骤 和 M-步骤 组成
        #E-步骤：计算期望值zeta和gamma
        for t in range(col-1):
            #分母部分
            denominator=numpy.dot(numpy.dot(alpha[:,t],A)*B[:,O[t+1]],beta[:,t+1])
            for i in range(row):
                #分子部分以及zeta的值
                numerator=alpha[i,t]*A[i,:]*B[:,O[t+1]]*beta[:,t+1]
                zeta[i,:,t]=numerator/denominator
        gamma=numpy.sum(zeta,axis=1)
        final_numerator=(alpha[:,col-1]*beta[:,col-1]).reshape(-1,1)
        final=final_numerator/numpy.sum(final_numerator)
        gamma=numpy.hstack((gamma,final))
        #M-步骤：重新估计参数Pi,A,B
        newPi=gamma[:,0]
        newA=numpy.sum(zeta,axis=2)/numpy.sum(gamma[:,:-1],axis=1)
        newB=numpy.copy(B)
        b_denominator=numpy.sum(gamma,axis=1)
        temp_matrix=numpy.zeros((1,len(O)))
        for k in range(B.shape[1]):
            for t in range(len(O)):
                if O[t]==k:
                    temp_matrix[0][t]=1
            newB[:,k]=numpy.sum(gamma*temp_matrix,axis=1)/b_denominator
        #终止阀值
        # if numpy.max(abs(Pi-newPi))<e and numpy.max(abs(A-newA))<e and numpy.max(abs(B-newB))<e:
        #     done=True 
        done=True
        A=newA
        B=newB
        Pi=newPi
    return A,B,Pi

emission_probability= [[ 0.6, 0.4, 0, 0 , 0 , 0 , 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0.2, 0.4 , 0.4 , 0 , 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0 , 0 , 0.5 , 0.3, 0.2, 0, 0, 0, 0, 0, 0],
                               [ 0, 0, 0, 0 , 0 , 0 , 0, 0, 0.1, 0.4 , 0.5, 0, 0, 0],
                               [0, 0, 0, 0 , 0 , 0 , 0, 0, 0, 0, 0, 0.3, 0.4 , 0.3]]
trainsition_probability= [[0 , 0, 0.5, 0 , 0.5],
                            [1, 0, 0, 0 , 0],
                            [0 , 0, 0.3, 0 , 0.7],
                            [0 , 1, 0, 0 , 0],
                            [0 , 0.5, 0, 0.5 , 0]]
pi = [0.2,0.2,0.2,0.2,0.2]
trainsition_probability = np.array(trainsition_probability)
emission_probability  = np.array(emission_probability)
    # print(emission_probability[:,0])
pi = np.array(pi)
obs_seq = [1,5,11,8,2,0,12]
A,B,Pi=baum_welch(trainsition_probability,emission_probability,pi,obs_seq)
print(A)
print(B)
print(Pi)