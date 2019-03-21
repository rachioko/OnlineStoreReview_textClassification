import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")

file = r'''C:\databases\Womensclothingreviews.csv'''
data = pd.read_csv(file)

#dataset information
print (data.shape)
print()
print (data.dtypes)
print()
print (data.head(5))
print()

#new data 
ndata = data[['Age','Rating','Recommended IND','Division Name','Department Name','Class Name']]
ndata.columns = ['Age', 'Rating', 'Recommended', 'Div_name','Department','class']
print (ndata.describe().transpose())
print()

#missing data
total = ndata.isnull().sum().sort_values(ascending=False)
print ('Total missing values:\n',total)


#Categorical information
var = 'Div_name'
C_data = ndata[var].value_counts()
print(var + ' column information:\n',C_data)
print()

sns.barplot(C_data.index, C_data.values, alpha=0.9)
plt.title('Frequency Distribution of clothing ' + var)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel(var, fontsize=12)
plt.show()

var = 'Department'
C_data = ndata[var].value_counts()
print(var + ' column information:\n',C_data)
print()
plt.subplots(figsize=(16, 8))
sns.barplot(C_data.index, C_data.values, alpha=0.9)
plt.title('Frequency Distribution of clothing ' + var)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel(var, fontsize=12)
plt.show()

var = 'class'
C_data = ndata[var].value_counts()
print(var + ' column information:\n',C_data)
print()
plt.subplots(figsize=(16, 8))
sns.barplot(C_data.index, C_data.values, alpha=0.9)
plt.title('Frequency Distribution of clothing ' + var)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel(var, fontsize=12)
plt.show()


#Age distribution graph
plt.subplots(figsize=(12, 5))
fig2 = sns.distplot(ndata['Age'])
plt.show() 
plt.subplots(figsize=(12, 5))
fig3= sns.boxplot(x="Department", y="Rating",hue= 'Recommended',data=ndata)
plt.ylim(0,6)


var = 'Rating'
sc = pd.concat([ndata['Age'], ndata[var]], axis=1)
f, ax = plt.subplots(figsize=(18, 9))
fig = sns.boxplot(y=var, x="Age", data=sc)
fig.axis(xmin=0, xmax=100);

#sc.plot.scatter(x=var, y='department',xlim=(0,5));

sns.pairplot(ndata[['Age', 'Rating', 'Recommended']], size = 2.5)
plt.show();
