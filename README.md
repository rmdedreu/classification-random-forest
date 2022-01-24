# classification-random-forest
This contains a step by step procedure of predicting 1 and 0 using random forest classification

import pandas as pd
import numpy as np

df = pd.read_csv('RetailCustomerSales.csv', sep = ';')
df.head(10)

df.info()

df.describe()

# Get a feeling for number of unique values per column
df.nunique()

# Category 2 and 3 have many missing values, fill with 0 
df['ItemCategory2'] = df['ItemCategory2'].fillna(0)
df['ItemCategory3'] = df['ItemCategory3'].fillna(0)
df.info()

# Encode string values to numerical 
city_dict = {'A': 1 , 'B': 2, 'C': 3}
sex_dict = {'F': 0 , 'M': 1}
age_dict = {'0-17': 0 ,'26-35': 2, '46-50': 4 , '51-55': 5, '36-45': 3, '18-25': 1, '55+': 6}
years_in_city_dict = {'4+': 4}
df['CityType'] = df['CityType'].replace(city_dict)
df['Sex'] = df['Sex'].replace(sex_dict)
df['Age'] = df['Age'].replace(age_dict)
df['YearsInCity'] = df['YearsInCity'].replace(years_in_city_dict)

# Correct the datatypes 
df = df.astype({  'Sex': 'int32',
                  'Age': 'int32', 
                  'Profession': 'int32',
                  'CityType': 'int32',
                  'YearsInCity': 'int32',
                  'ItemCategory1': 'int32', 
                  'ItemCategory2': 'int32',
                  'ItemCategory3': 'int32'})
df.info()

#Create DF with missing values for have_children


missing_values_df = df[df['HaveChildren'].isna()].copy()



missing_values_df.info()

# Drop rows with missing values from dataset 
df = df.dropna(how='any')
df.info()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# Initialize model 
model_rf = RandomForestClassifier(n_estimators = 100, max_depth = 100, random_state = 42)

# define features
features = ['Sex'          ,
            'Age'          ,
            'Profession'   ,
            'CityType'     ,
            'YearsInCity'  ,
            'ItemCategory1',
            'ItemCategory2', 
            'ItemCategory3']
            
# Perform train test split 
X_train, X_test, y_train, y_test = train_test_split(df[features], df['HaveChildren'], test_size=0.33, random_state=42)

# Fit the model 
model_rf.fit(X_train, y_train)

# Predict using the fitted model
y_pred = model_rf.predict(X_test)

print("recall_score (Random Forest): ", recall_score(y_pred, y_test))
print("precision_score (Random Forest): ", precision_score(y_pred, y_test))
print("f1_score (Random Forest): ", f1_score(y_pred, y_test))

## Model_rf predict for missing values
missing_values_df = missing_values_df[features]


missing_values_predict =model_rf.predict(missing_values_df)


pd.DataFrame(missing_values_predict).to_csv("predicted missing values.csv")


