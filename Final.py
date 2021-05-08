import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('Dallas_Animals_Field_Data.csv')
pd.set_option('display.max_columns',None)

#check N/A value in this data set
print(df[df.isnull().any(axis=1)])

print('Total Feral Animals in Dallas:',len(df))

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept = True)

print('\nDallas Feral Animals Status:')
#df_animals_status = df['Activity Type'].value_counts()
print(df['Activity Type'].value_counts().to_string(),'\n')

#first figure
f1 = plt.figure(1)
df['Activity Type'].value_counts().plot.bar()

plt.title('Dallas Feral Animals Status')
plt.ylabel('Count Numbers')
plt.xlabel('Status Types')

#second figure
f2 = plt.figure(2)
#count activity total
print('Activity Priority')
print(df['Activity Priority'].value_counts().to_string(),'\n')

print('Count of Animals')
#count animal type total
print(df['Animal Type'].value_counts().to_string(),'\n')

df['Animal Type'].value_counts().plot.bar()
plt.title('Missing Animal Types')
plt.ylabel('Total Count')
plt.xlabel('Pet Type')

#showing status of these animals
print('Status')
print(df['Activity Result 1'].value_counts().to_string(),'\n')
df1 = df['Activity Result 1'].value_counts()

#results
print('Results')
print(df['Activity Status'].value_counts().to_string(),'\n')

#third figure
print('Loosing Animals for Each Month')
print(df['Month'].value_counts().to_string(),'\n')

new_df = df.groupby('Month').count().reset_index().rename(columns={'Month': 'Month'})
new_df.plot.scatter(x='Month', y='Activity Number', title= "Actvity Animals Numbers for The Month")

plt.show()
# liner regression
# xNoCat = 
#x3 = pd.get_dummies(df['dog'])
#y3 = new_df['Activity Number']
#model.fit(x3,y3)
#
#coefResults = list(zip(x.columns, model.coef_))
#for coefResult in coefResults:
#    print(str(coefResult[0]).ljust(30)," ",str(coefResult[1]).rjust(25))
