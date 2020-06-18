import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as XGB

data = pd.read_csv("Train.csv")

X = data.drop(['INCIDENT_ID','DATE','MULTIPLE_OFFENSE'], axis=1)
Y = data['MULTIPLE_OFFENSE']

x = X.values
y = Y.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

clf = XGB(seed=0)
clf.fit(x_train, y_train)
# rfc = RandomForestClassifier()
# rfc.fit(x_train,y_train)

# y_pred = rfc.predict(x_test)

# from sklearn.metrics import confusion_matrix 
# import matplotlib.pyplot as plt
# import seaborn as sns

# LABELS = ['Normal', 'Fraud'] 
# conf_matrix = confusion_matrix(y_test, y_pred) 
# plt.figure(figsize =(12, 12)) 
# sns.heatmap(conf_matrix, xticklabels = LABELS,  
#             yticklabels = LABELS, annot = True, fmt ="d"); 
# plt.title("Confusion matrix") 
# plt.ylabel('True class') 
# plt.xlabel('Predicted class') 
# plt.show() 

test_data = pd.read_csv("Test.csv")
filenames = test_data['INCIDENT_ID'];
test_data = test_data.drop(['INCIDENT_ID','DATE'], axis=1)

test_values = test_data.values

test_pred = clf.predict(test_values)

results=pd.DataFrame({"INCIDENT_ID":filenames,
                      "MULTIPLE_OFFENSE":test_pred})
results.to_csv("result.csv",index=False)