# EXNO:4-DS
# Name : M HARSHITH
# REG NO : 212224040206
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv")
data

<img width="1776" height="906" alt="image" src="https://github.com/user-attachments/assets/dee4b5f4-1e4c-4b3e-aa6f-0685802d38a8" />

data.isnull().sum()

<img width="367" height="663" alt="image" src="https://github.com/user-attachments/assets/6f4b9255-3b14-4110-91b4-03ad819e391b" />

missing=data[data.isnull().any(axis=1)]
missing

<img width="1415" height="167" alt="image" src="https://github.com/user-attachments/assets/2ccb8ab7-d669-490d-a8b7-3f163f13ba04" />

data2=data.dropna(axis=0)
data2

<img width="1736" height="808" alt="Screenshot 2025-09-30 114354" src="https://github.com/user-attachments/assets/b349833e-7276-42aa-85d4-33419968e7a0" />

sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

<img width="1062" height="377" alt="image" src="https://github.com/user-attachments/assets/c13ee527-32b6-450a-90ea-49449422aecb" />

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs

<img width="558" height="623" alt="image" src="https://github.com/user-attachments/assets/fde976e1-d5af-4584-9d2c-2e92c333819e" />

data2

<img width="1588" height="576" alt="image" src="https://github.com/user-attachments/assets/f4c1f057-c0df-4097-899a-fd606ec2b6bd" />

new_data=pd.get_dummies(data2, drop_first=True)
new_data

<img width="1777" height="673" alt="image" src="https://github.com/user-attachments/assets/8800fd9a-f203-4961-afa0-2e2edf630219" />

columns_list=list(new_data.columns)
print(columns_list)

<img width="1775" height="130" alt="image" src="https://github.com/user-attachments/assets/1a8d8db6-ef49-4387-a7c8-bfe5b7ecbe85" />

features=list(set(columns_list)-set(['SalStat']))
print(features)

<img width="1783" height="122" alt="image" src="https://github.com/user-attachments/assets/dc969abe-1e60-46b4-9334-041a84e75501" />

y=new_data['SalStat'].values
print(y)

<img width="412" height="116" alt="image" src="https://github.com/user-attachments/assets/c2ee27a3-a921-4a0f-acab-56a5ab967702" />

x=new_data[features].values
print(x)

<img width="508" height="242" alt="image" src="https://github.com/user-attachments/assets/d74fe164-b073-4af3-a9ca-a343bb4a4f2e" />

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

<img width="878" height="197" alt="image" src="https://github.com/user-attachments/assets/b4324809-cf8c-4161-96c3-720f5f3c0272" />

prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

<img width="597" height="185" alt="image" src="https://github.com/user-attachments/assets/f0720617-1111-4e43-b836-8ec885e42df1" />

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

<img width="571" height="111" alt="image" src="https://github.com/user-attachments/assets/7843f128-6302-4c13-aa22-9646062910fd" />

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

<img width="767" height="92" alt="image" src="https://github.com/user-attachments/assets/206eeea5-ded1-4b55-8ee3-1b65808e937e" />

data.shape

<img width="467" height="95" alt="image" src="https://github.com/user-attachments/assets/0de567c2-be33-45e2-9da4-be00109fbf9a" />

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

<img width="1781" height="575" alt="image" src="https://github.com/user-attachments/assets/6be4fa56-49b5-4f80-a7cc-52df2c536fd2" />

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

<img width="837" height="475" alt="image" src="https://github.com/user-attachments/assets/eebd37d3-9c20-47c4-a701-5c5a1218fdb7" />

tips.time.unique()

<img width="532" height="112" alt="image" src="https://github.com/user-attachments/assets/105cf46c-c3f5-4e6d-adf5-4838cb47d8a0" />

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

<img width="663" height="178" alt="image" src="https://github.com/user-attachments/assets/0df7929a-3394-4dd2-bdc7-d227d5b6764e" />

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

<img width="632" height="158" alt="image" src="https://github.com/user-attachments/assets/91537a8e-abcd-4600-887d-546a769dbf97" />


# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.
