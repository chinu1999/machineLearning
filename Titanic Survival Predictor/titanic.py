import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

df['Family_Size']=df['SibSp']+df['Parch']
df2['Family_Size']=df2['SibSp']+df2['Parch']

df['Age'].fillna(0 , inplace = True)
df2['Age'].fillna(0 , inplace = True)

def age_classification(age):
    
    if age >= 1 and age < 15:
        return 1
    elif age >= 15 and age <= 30:
        return 2
    elif age > 30  and age <= 50:
        return 3
    elif age > 50  and age <= 70:
        return 4
    elif age > 70  and age <= 90:
        return 5
    return 0

df['age_class'] = df['Age'].apply(lambda x :  age_classification(int(x)))
df2['age_class'] = df2['Age'].apply(lambda x :  age_classification(int(x)))

from sklearn.preprocessing import LabelEncoder
label_sex = LabelEncoder()
label_cabin = LabelEncoder()

df['gender'] = label_sex.fit_transform(df['Sex'])
df2['gender'] = label_sex.fit_transform(df2['Sex'])

X_train = pd.DataFrame({
    'pclass' : df['Pclass'],
    'age' : df['age_class'],
    'family' : df['Family_Size'],
    'gender' : df['gender'],
    'survived' : df['Survived']
})
    
X_test = pd.DataFrame({
    'pclass' : df2['Pclass'],
    'age' : df2['age_class'],
    'family' : df2['Family_Size'],
    'gender' : df2['gender']
})

Y_train = X_train['survived']
    
X_train = X_train.drop('survived',axis=1)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, Y_train)
final_preds=clf.predict(X_test)
    
submission = pd.DataFrame({
        "PassengerId":df2["PassengerId"],
        "Survived": final_preds
    })
submission.to_csv('submission.csv', index=False)