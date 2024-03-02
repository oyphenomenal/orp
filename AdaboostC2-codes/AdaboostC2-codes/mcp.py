from AdaboostC2 import readData_CV2
from AdaboostC2 import AdaboostC3
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression as LR
from mReadData import *
from mEvaluation import evaluate
from operator import itemgetter
import random
import pandas as pd
from mEvaluation import evaluate
from scipy.stats import chi2
from scipy.stats import ncx2
from adacost import AdacostC3
from adacost import readData_CV2
import pprint
pd.set_option('display.max_rows', 20)
'''
Hosmer_Lemeshow_testcg，它用于执行Hosmer-Lemeshow拟合度检验。
该方法根据给定的数据进行计算，并返回拟合度统计量（HLtest）和对应的p值。
在计算HLtest时，首先对数据进行排序，然后将其分成Q组。
接着根据每组的真实标签和预测概率计算相关的统计量，最后计算HLtest值。
然后，根据HLtest值和自由度Q-2（组数减去参数数量）计算p值。
'''

class montecar (AdacostC3):

    def __init__(self, Q, T, delta=0.01):
        super().__init__(Q, T, delta)
 

    def Hosmer_Lemeshow_testcg(self,data, Q=79):
                                                                            
        data = data.sort_values('y_hat')
        # print(data)
        data['Q_group'] = pd.qcut(data['y_hat'], Q,duplicates='drop')
        # print(data['Q_group'])
        y_p = data['y'].groupby(data.Q_group).sum()
        y_total = data['y'].groupby(data.Q_group).count()
        y_n = y_total - y_p
        # print(y_p)
        y_hat_p = data['y_hat'].groupby(data.Q_group).sum()
        y_hat_total = data['y_hat'].groupby(data.Q_group).count()
        y_hat_n = y_hat_total - y_hat_p
        # print(pd.concat([y_p,y_n,y_hat_p,y_hat_n],axis=1))
        hltest = (((y_p - y_hat_p)**2 / y_hat_p) + ((y_n - y_hat_n)**2 / y_hat_n)).sum()
        l = hltest-(Q-2)
        if l<0:
            pval = 1-chi2.cdf(hltest,Q-2)
        else:
            pval = 1-ncx2.cdf(hltest,Q-2,hltest-(Q-2))       
        # print(hltest)
        return hltest ,pval


