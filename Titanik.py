from inspect import Parameter
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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

#Преобразования
# 0 - Unknown, 1 - Cherbourg, 2 - Queenstown, 3 - Southampton
dft[age] = dft[age].fillna(0)
dft[embarked] = dft[embarked].fillna(0)
dft[sex] = dft[sex].map(lambda sex: int(sex == 'male'))
dft[embarked] = dft[embarked].replace({'C':int(1),'Q':int(2),'S':int(3)})

dftest[age] = dftest[age].fillna(0)
dftest[fare] = dftest[fare].fillna(0)
dftest[embarked] = dftest[embarked].fillna(0)
dftest[sex] = dftest[sex].map(lambda sex: int(sex == 'male'))
dftest[embarked] = dftest[embarked].replace({'C':int(1),'Q':int(2),'S':int(3)})

#Новый CVS
dft = dft.drop(['Survived','PassengerId','Name','Ticket','Cabin'], axis=1)
dftest = dftest.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
print(dftest[:LastShow])
print(dftest.info())

#ML
tree = DecisionTreeClassifier(max_depth=2, random_state=17)
tree.fit(dft, sur)
predictions = tree.predict(dftest)
print(predictions)
#Вывод
