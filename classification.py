import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import sys
plt.style.use("ggplot")

parentDir = os.path.dirname(os.getcwd())
currentDir = parentDir + "/sentimentAnalyser/"
feature = currentDir + "training_features.txt"
feature_output = currentDir + "training_features_output.txt"

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")
X_train = []
X_test = []
Y_train = []
Y_test = []
i = 0
with open(feature, 'rb') as f:
    file_content = f.read().splitlines()
    for i,line in enumerate(file_content):
        lst = [float(x) for x in line[1: -1].split(',')]
        # if (i < 3700):
        X_train.append(lst)
            # print line
            # print type(line)
        # elif (i >= 5331 and i < 9031):
        #     X_train.append(lst)
        # else:
        #     X_test.append(lst)

with open(feature_output, 'rb') as f:
    file_content = f.read().splitlines()
    for line in file_content:
        # print line
        # lst = [float(x) for x in line[1: -1].split(',')]
        Y_train.append(float(line))


# for i in range(3262):
#     if (i < 1631):
#         Y_test.append(0)
#     else:
#         Y_test.append(1)
clf1 = svm.SVC()
grid = GridSearchCV(clf1, param_grid={'C': [1, 10]})


clf2 = MultinomialNB()
clf3 = LogisticRegression()
# clf2 = svm.SVC(kernel='linear', C=0.8)
#
# scores1 = cross_val_score(clf1, X_train, Y_train, cv=5)
# scores2 = cross_val_score(clf2, X_train, Y_train, cv=5)
# scores3 = cross_val_score(clf3, X_train, Y_train, cv=5)
# scores4 = cross_val_score(clf2, X_train, Y_train, cv=5)
#
# print scores1
# print scores2
# print scores3
# print scores4
# clf1.fit(X_train, Y_train)
# clf2.fit(X_train, Y_train)
# pred1 = clf1.predict(X_test)
# pred2 = clf2.predict(X_test)

pred1 = cross_val_predict(clf1, X_train, Y_train, cv=2)
pred2 = cross_val_predict(clf2, X_train, Y_train, cv=2)
pred3 = cross_val_predict(clf3, X_train, Y_train, cv=2)

Accuracy1 = accuracy_score(Y_train, pred1)
Accuracy2 = accuracy_score(Y_train, pred2)
Accuracy3 = accuracy_score(Y_train, pred3)



cm1 = confusion_matrix(pred1, Y_train)
cm2 = confusion_matrix(pred2, Y_train)
cm3 = confusion_matrix(pred3, Y_train)
# print (clf1.score(X_test, Y_test))
# print (cm1)
# print (clf2.score(X_test, Y_test))
# print (cm2)
print Accuracy1
print Accuracy2
print Accuracy3

print(classification_report(Y_train, pred1, target_names=["Negative", "Postive"]))
print(classification_report(Y_train, pred2, target_names=["Negative", "Postive"]))
print(classification_report(Y_train, pred3, target_names=["Negative", "Postive"]))

plt.matshow(cm1)
plt.title('Confusion matrix of the %s classifier' % "SVM")
plt.colorbar()
plt.show()
# plt.savefig('fig_SVM.png')

plt.matshow(cm2)
plt.title('Confusion matrix of the %s classifier' % "MultinomialBayes")
plt.colorbar()
plt.show()
# plt.savefig('fig_MB.png')


plt.matshow(cm3)
plt.title('Confusion matrix of the %s classifier' % "Maximum Entropy")
plt.colorbar()
plt.show()
# plt.savefig('fig_MaxEnt.png')

# pause()