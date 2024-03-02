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
from mcp import *

if __name__ == "__main__":

    k_fold, X_all, Y_all = readData_CV2()
    modelsave3 = 0
    modelpp = np.zeros([1, 5])
    for train, test in k_fold.split(X_all, Y_all):
        X = X_all[train]
        Y = Y_all[train]
        Xt = X_all[test]
        Yt = Y_all[test]
        Yt = pd.DataFrame(Yt)
        classifier = montecar(5, 10)
        classifier.induce(X, Y)
        import joblib

        joblib.dump(classifier, 'model.joblib')
        prediction = pd.DataFrame(classifier.test(Xt))

        h = np.zeros([1, 5])
        p = np.zeros([1, 5])
        for i in range(0, 5):
            data = pd.concat([prediction.iloc[:, i], Yt.iloc[:, i]], axis=1)
            data.columns = ['y_hat', 'y']
            h1, pp = classifier.Hosmer_Lemeshow_testcg(data)
            h[:, i] = h1
            p[:, i] = pp
        # h = h/5
        # print(h)
        # print(p)
        modelsave3 += h
        modelpp += p
        # print(p)
    import numpy as np
    from sklearn.metrics import label_ranking_loss

    # print(label_ranking_loss(Yt, prediction))
    # print("mean--",modelsave3/5)
    # print("mean--",modelpp/5)
    # pd.options.display.max_columns = 5

    prediction.to_csv(r'D:\PythonProjects\orp\AdaboostC2-codes\AdaboostC2-codes\pp.csv', index=False)
    Yt.to_csv(r'D:\PythonProjects\orp\AdaboostC2-codes\AdaboostC2-codes\yy.csv', index=False)
