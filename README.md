# EXNO:4-DS
# MOHAN RAJ.S
# 212224100036
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
~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
~~~
![image](https://github.com/user-attachments/assets/0e5b6ee1-43b3-4f40-a780-5f39abf4a7c8)
~~~
df_null_sum=df.isnull().sum()
df_null_sum
~~~
![image](https://github.com/user-attachments/assets/dd34256c-2922-49e8-bc2a-032eadec237f)
~~~
df.dropna()
~~~
![image](https://github.com/user-attachments/assets/1adf3460-df3e-45b2-861d-8094e93ba042)
~~~
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
~~~
![image](https://github.com/user-attachments/assets/151baa13-3d96-42dd-8b83-33152c7a5c4a)

# STANDARD SCALING:

~~~
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
~~~
![image](https://github.com/user-attachments/assets/d5bbf6fb-454c-463c-9f86-63920776db9c)
~~~
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
~~~
![image](https://github.com/user-attachments/assets/44c04a54-36f1-4b4d-b32b-0a9417bbe3d0)

# MIN-MAX SCALING:

~~~
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
~~~
![image](https://github.com/user-attachments/assets/0a364a7d-ca30-43a1-a543-84a91e702a74)

# MAXIMUM ABSOLUTE SCALING:

~~~
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
~~~
![image](https://github.com/user-attachments/assets/c95322bb-8bb8-4dc0-8154-708c3ea445e8)
~~~
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
~~~
![image](https://github.com/user-attachments/assets/ecc68d74-45b0-40ea-a40c-6ca669ca36dd)

# ROBUST SCALING:

~~~
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
~~~
![image](https://github.com/user-attachments/assets/801466fc-3232-4f69-8355-371cea52d83e)
~~~
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
~~~
![image](https://github.com/user-attachments/assets/091bca82-225b-47a5-b071-864bfdac14ce)

# FEATURE SELECTION:

~~~
import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
~~~
![image](https://github.com/user-attachments/assets/bdf1c05b-9e42-484f-a3c6-af4f5b43ee71)
~~~
df
~~~
![image](https://github.com/user-attachments/assets/14f83721-fce8-4d09-b53d-21d0b29233ae)
~~~
df.info()
~~~
![image](https://github.com/user-attachments/assets/99afd126-3f31-4ec5-8544-78df043b0f82)
~~~
df_null_sum=df.isnull().sum()
df_null_sum
~~~
![image](https://github.com/user-attachments/assets/0aff510d-89db-4270-9866-34c0a7a2eb98)
~~~
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
~~~
![image](https://github.com/user-attachments/assets/43365cdb-bbd8-4403-9cbc-34e5daf636ab)
~~~
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
~~~
![image](https://github.com/user-attachments/assets/41305456-f38a-488b-8c1a-64b21944eb10)
~~~
X = df.drop(columns=['SalStat'])
y = df['SalStat']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
~~~
![image](https://github.com/user-attachments/assets/30d866b6-50aa-45e1-842e-64a98d685463)
~~~
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
~~~
![image](https://github.com/user-attachments/assets/fb4a8de6-5f6b-4ea9-a392-00eb4baad904)
# Fisher Square:
~~~
!pip install skfeature-chappers
~~~
![image](https://github.com/user-attachments/assets/1b87e100-aff5-49ca-add1-aa2b1852666d)
~~~
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
~~~
~~~
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
~~~
~~~
df[categorical_columns]
~~~
![image](https://github.com/user-attachments/assets/b0f74f47-3782-4529-9c62-f858483a791e)
~~~
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
~~~
~~~
df[categorical_columns]
~~~
![image](https://github.com/user-attachments/assets/adc95d28-6b22-4181-81c9-71582a8c5409)

# ANOVA

~~~
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define X and y before using them
df=pd.read_csv("/content/income(1) (1).csv") # Assuming the data is in this file, make sure to adjust if necessary
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
X = df.drop(columns=['SalStat'])  # Define X by dropping the target variable
y = df['SalStat']  # Define y as the target variable

k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova) 
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
~~~
![image](https://github.com/user-attachments/assets/cc5c0e24-a549-41d8-94b5-5a0c915a4e75)

#  WRAPPER METHOD:

~~~
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
~~~
~~~
df[categorical_columns]
~~~
![image](https://github.com/user-attachments/assets/50b5ae10-a64c-434f-92e6-72ae5866882f)
~~~
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)

df[categorical_columns]
~~~
![image](https://github.com/user-attachments/assets/6882307b-e563-481f-93d6-e9f8d2314440)
~~~
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
~~~
![image](https://github.com/user-attachments/assets/34060fcb-aada-484c-bae9-9f6b3d251081)
~~~
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
~~~
![image](https://github.com/user-attachments/assets/49b5d1bf-4751-4101-9ca3-2743b4e8283e)
~~~
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
~~~
![image](https://github.com/user-attachments/assets/56c2ef66-6513-4604-86e0-16743eb6cdde)

# RESULT:
       Feature Scaling and Feature Selection is successfully implemented.
