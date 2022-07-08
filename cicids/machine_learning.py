import torchvision
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from time import time
from dataload import X_train, X_test, y_train, y_test


"""
DecisionTree
"""
start = time()
clf = tree.DecisionTreeClassifier() 
clf = clf.fit(X_train,y_train) 
result = clf.score(X_test,y_test)
end = time()

print(f'DT:')
print(f'acc:{result}')
print(f'time:{end-start}')


'''
NB
'''
start = time()

gnb = GaussianNB()
gnb = gnb.fit(X_train,y_train)
result = gnb.score(X_test,y_test)
end = time()

print(f'NB:')
print(f'acc:{result}')
print(f'time:{end-start}')
'''
LR
'''
start = time()

LR = LogisticRegression()
LR = LR.fit(X_train,y_train)
result = LR.score(X_test,y_test)
end = time()

print(f'LR:')
print(f'acc:{result}')
print(f'time:{end-start}')

'''
KNN
'''
start = time()

knn = KNeighborsClassifier(3)
knn = knn.fit(X_train,y_train)
result = knn.score(X_test,y_test)
end = time()

print(f'KNN:')
print(f'acc:{result}')
print(f'time:{end-start}')

'''
SVM
'''
print(len(X_test))
start = time()

svc = SVC()
svc = svc.fit(X_train[:60000],y_train[:60000])
result = svc.score(X_test,y_test)

end = time()

print(f'SVM:')
print(f'acc:{result}')
print(f'time:{end-start}')

'''
RF
'''

start = time()

rf = RandomForestClassifier()
rf = rf.fit(X_train,y_train)
result = rf.score(X_test,y_test)

end = time()

print(f'rf:')
print(f'acc:{result}')
print(f'time:{end-start}')

