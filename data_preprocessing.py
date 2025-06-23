#import the necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os

#Load the Dataset
df = pd.read_csv("Titanic-Dataset.csv")
df.head()

#Basic Exploration
df.info()
df.describe()
df.isnull().sum()

#Handle Missing Values
df['Age'] = (df['Age'].fillna(df['Age'].median()))

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop(columns=['Cabin'], inplace = True)

#Dropping Irrelevant Columns
df.drop(columns=['Name','Ticket'],inplace=True)

#Encode Categorical Variables
df['Sex'] = df['Sex'].map({'male' : 0, 'female' : 1})

df = pd.get_dummies(df, columns=['Embarked'],drop_first = True)

#Normalize Numerical features
scaler = StandardScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])

#Visualising the data
#sns.boxplot(x=df['Fare'])
#plt.show()

#removing Outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower_bound ) & (df['Fare'] <= upper_bound)]

#Final check
df.isnull().sum()
df.head()

#save the cleaned dataset
df.to_csv("cleaned_titanic.csv",index = False)
print("✅ File saved at:", os.path.abspath("cleaned_titanic.csv"))