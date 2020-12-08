import numpy as np
import pandas as pd
train=pd.read_csv("Data/train.csv")
train.SibSp.unique()
quali=[]
quanti=[]
for i in train.columns:
    if train[i].dtypes=='object':
        quali.append(i)
    else:
        quanti.append(i)
print("Qualitative:",len(quali))
print("Quantitative:",len(quanti))
for i in train.columns:
    print(i, (train[i].isna().sum())/(train.shape[0]))
    train.drop(columns='Cabin',axis=1,inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
train['Age'].fillna(train['Age'].mean(),inplace=True)
train.isna().sum()
train['Pclass']=train['Pclass'].astype("object")
train.info()
quali=[]
quanti=[]
for i in train.columns:
    if train[i].dtypes=='object':
        quali.append(i)
    else:
        quanti.append(i)
print("Qualitative:",len(quali))
print("Quantitative:",len(quanti))

def parse_model(X,use_columns):
    if "Survived" not in X.columns:
        raise ValueError('target column survived should belong to df')
    target=X['Survived']
    X=X[use_columns]
    return X,target
modelcols1=['SibSp','Parch','Fare']
X,y=parse_model(X=train.copy(),use_columns=modelcols1)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0,)
from sklearn.metrics import classification_report
lr=LogisticRegression()
lr.fit(X_train,y_train)
print("Train :",lr.score(X_train,y_train))
print("Test :",lr.score(X_test,y_test))
y_pred=lr.predict(X_test)
print(classification_report(y_test, y_pred))
clf=LogisticRegression()
print(cross_val_score(clf,X,y,cv=5).mean())
survivied=train[train.Survived==1]
dead=train[train.Survived==0]
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
%matplotlib inline
%pylab inline

def plot_hist(feature,bins=20):
    x1=np.array(dead[feature].dropna())
    x2=np.array(survivied[feature].dropna())
    plt.hist([x1,x2],label=["Victime","Survivant"],bins=bins,color=['r','b'])
    plt.legend(loc="upper left")
    plt.title('distribution relative de %s' %feature)
    plt.show()
plot_hist('Pclass')
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)
lr=LogisticRegression()
lr.fit(X_train,y_train)
print("Train :",lr.score(X_train,y_train))
print("Test :",lr.score(X_test,y_test))
y_pred=lr.predict(X_test)
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
%matplotlib inline
%pylab inline

def plot_hist(feature,bins=20):
    x1=np.array(dead[feature].dropna())
    x2=np.array(survivied[feature].dropna())
    plt.hist([x1,x2],label=["Victime","Survivant"],bins=bins,color=['r','b'])
    plt.legend(loc="upper left")
    plt.title('distribution relative de %s' %feature)
    plt.show()
plot_hist('Age')
modelcols1=['SibSp','Parch','Fare','Pclass','Age']

X,y=parse_model(X=train.copy(),use_columns=modelcols1)
X=pd.get_dummies(X,columns=['Pclass'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)
lr=LogisticRegression()
lr.fit(X_train,y_train)
print("Train :",lr.score(X_train,y_train))
print("Test :",lr.score(X_test,y_test))
y_pred=lr.predict(X_test)
print(classification_report(y_test, y_pred))
modelcols1=['SibSp','Parch','Fare','Pclass','Age','Sex']

X,y=parse_model(X=train.copy(),use_columns=modelcols1)
X=pd.get_dummies(X,columns=['Pclass','Sex'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)
lr=LogisticRegression()
lr.fit(X_train,y_train)

print("Train :",lr.score(X_train,y_train))
print("Test :",lr.score(X_test,y_test))
y_pred=lr.predict(X_test)
print(classification_report(y_test, y_pred))
train.Age=train['Age'].apply(lambda x: 0.0 if (x>0 and x<10) else 1)
train.Age.unique()
modelcols1=['SibSp','Parch','Fare','Pclass','Age']

X,y=parse_model(X=train.copy(),use_columns=modelcols1)
X=pd.get_dummies(X,columns=['Pclass'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)
lr=LogisticRegression()
lr.fit(X_train,y_train)
print("Train :",lr.score(X_train,y_train))
print("Test :",lr.score(X_test,y_test))
y_pred=lr.predict(X_test)
print(classification_report(y_test, y_pred))
modelcols1=['SibSp','Parch','Fare','Pclass','Age','Sex']

X,y=parse_model(X=train.copy(),use_columns=modelcols1)
X=pd.get_dummies(X,columns=['Pclass','Sex'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)
lr=LogisticRegression()
lr.fit(X_train,y_train)
print("Train :",lr.score(X_train,y_train))
print("Test :",lr.score(X_test,y_test))
y_pred=lr.predict(X_test)
print(classification_report(y_test, y_pred))
train.Name[1].split()[1]
train['Titre']=train['Name'].apply(lambda x: x.split()[1])
train.Titre.unique()
train.info()
modelcols1=['SibSp','Parch','Fare','Pclass','Age','Sex','Titre']

X,y=parse_model(X=train.copy(),use_columns=modelcols1)
X=pd.get_dummies(X,columns=['Pclass','Sex','Titre'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)
lr=LogisticRegression()
lr.fit(X_train,y_train)
print("Train :",lr.score(X_train,y_train))
print("Test :",lr.score(X_test,y_test))
y_pred=lr.predict(X_test)
print(classification_report(y_test, y_pred))
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42,n_estimators=100,criterion="gini",max_depth=20)
model.fit(X_train,y_train)
model.score(X_test,y_test)
Estimators=RandomForestClassifier(random_state=42)
parameters={
    'n_estimators':[100,150,200,250,300],
    'max_depth':np.arange(6,16,2),
    'min_samples_split':np.arange(10,30,5)
}

from sklearn.model_selection import GridSearchCV
model2=GridSearchCV(Estimators,parameters,verbose=1,cv=5,n_jobs=-1)
model2.fit(X_train,y_train)
model2.best_params_
model3=RandomForestClassifier(random_state=42,n_estimators=300,criterion="gini",max_depth=6,min_samples_split=10)
model3.fit(X_train,y_train)
model3.score(X_test,y_test)