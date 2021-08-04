import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

#Константы
index = 0
name = "Name"
sex = "Sex"
pclass = "PClass"
age = "Age"
survived = "Survived"
LastShow = 15
embarked = "Embarked"
fare = "Fare"
dftest = pd.read_csv('test.csv')  #Test
dft = pd.read_csv('train.csv')    #Learn
sur = dft['Survived']
target = dft.Survived
result = pd.DataFrame(dftest.PassengerId)

#Преобразования
# 0 - Unknown, 1 - Cherbourg, 2 - Queenstown, 3 - Southampton
dft[age] = dft[age].fillna(0)
dft[embarked] = dft[embarked].fillna(dft[embarked].value_counts().index[0])
dft[sex] = dft[sex].map(lambda sex: int(sex == 'male'))
dft[embarked] = dft[embarked].replace({'C':int(1),'Q':int(2),'S':int(3)})

dftest[age] = dftest[age].fillna(0)
dftest[fare] = dftest[fare].fillna(0)
dftest[embarked] = dftest[embarked].fillna(dftest[embarked].value_counts().index[0])
dftest[sex] = dftest[sex].map(lambda sex: int(sex == 'male'))
dftest[embarked] = dftest[embarked].replace({'C':int(1),'Q':int(2),'S':int(3)})

#Новый CVS
dft = dft.drop(['Survived','PassengerId','Name','Ticket','Cabin'], axis=1)
dftest = dftest.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

#ML
clf = RandomForestClassifier(criterion='entropy',
    n_estimators=700,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features='auto',
    oob_score=True,
    random_state=1,
    n_jobs=-1)
x_train, x_test, y_train, y_test = train_test_split(dft, sur, test_size=0.25, random_state=2)
clf.fit(x_train, np.ravel(y_train))
proba = clf.predict_proba(x_test)
proba_true = proba[:,1]
fpr , tpr , treshold = roc_curve(y_test, proba_true)

#Вывод
au = roc_auc_score(y_test, proba_true)
print('LogisticRegression: ROC AUC=%.3f' % (au))

roc_auc = auc(fpr, tpr) # ROC анализ
print('ROC AUC: %0.2f' % roc_auc)
result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy') # Проверка скользящего контроля
print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2), "%")

clf.fit(dft, target)
result.insert(1,'Survived', clf.predict(dftest))
result.to_csv('result.csv', index=False)