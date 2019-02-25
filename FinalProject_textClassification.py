
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
#nltk.download('stopwords') 
from nltk.corpus import stopwords

file = r'''C:\Users\rachi\Documents\UCLA\Data Science\Projects\datasets\Womensclothingreviews.csv'''
data = pd.read_csv(file)

#data Information rows and columns
#print(data.shape)

#print (data.head())
#print(data.dtypes)
#print (data.info())

#new data 
ndata = data[['Review Text','Rating','Recommended IND']]
print (ndata.describe().transpose())
print()

#check for empty text reviews and reduce features
emptyColumns = ndata.columns[ndata.isnull().any()]
print(ndata[emptyColumns].isnull().sum())
emptyReview = ndata[ndata["Review Text"].isnull()][emptyColumns]
TextData = ndata.drop(emptyReview.index) #data without empty text reviews  
print (TextData.info())


#add new column text length of the reviews
#TextData['textlength'] = TextData['Review Text'].apply(len)
#print(TextData.head(10))

#Plot Visualization of rating vs text length distribution 
#pt1 = sns.FacetGrid(data=TextData, col='Rating')
#pt1.map(plt.hist, 'textlength', bins=50)
#plt.show(pt1)

#Plot Visualization of rating vs text length distribution 
#pt2 = sns.FacetGrid(data=TextData, col='Recommended IND')
#pt2.map(plt.hist, 'textlength', bins=50)
#plt.show(pt2)

#pt3 = sns.boxplot(x='Rating', y='textlength', data=TextData)
#plt.show(pt3)

#plt.subplots(figsize=(16, 8))
#sns.barplot(x='Rating', y='textlength', data=TextData)
#plt.title('Frequency Distribution')
#plt.ylabel('Text Length', fontsize=12)
#plt.xlabel('Rating', fontsize=12)
#plt.show()


#rel = TextData.groupby('Rating').mean()
#print(rel.corr())
#pt4 = sns.heatmap(data=rel.corr(), annot=True)
#plt.show(pt4)

#get variables for prediction for ratings 1 and 5
#predict_data = TextData[(TextData['Rating']==1) | (TextData['Rating']==5)]

#get variables for prediction for Recommended
predict_data = TextData
#print (predict_data.shape)

x = predict_data['Review Text']
y = predict_data['Recommended IND']

samplereview = x.iat[8989]
print(samplereview)

#show value 
#print(x.iat[0])

#get feature vector for the classification task and use bag of words for corpus

import string
def text_processing(text):
# Takes in a string of text, Removes all punctuation and stopwords    
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#Test of the processing
sample = "The Long Night is nearly upon us!! Winter is Coming."
#print(text_processing(sample))

#Using CountVectorizer to change text to vectors as a 2d matrix[column=reviewText, rows= uniqueWords]
#BagOfWordsMatrix

import sklearn
from sklearn.feature_extraction.text import CountVectorizer  
#BOfW = CountVectorizer(analyzer = text_processing).fit(x)
#print(len(BOfW.vocabulary_))

#samplereview = x.iat[8989]
#print(samplereview)
#samplebofw= BOfW.transform([samplereview])
#print(samplebofw)

#print(BOfW.get_feature_names()[13818])
#print(BOfW.get_feature_names()[15634])
#print(BOfW.get_feature_names()[19092])

#transform x into sparse matrix
#X = BOfW.transform(x)
#print('Shape of Sparse Matrix: ', X.shape)
#print('Amount of Non-Zero occurrences: ', X.nnz)

# Percentage of non-zero values
#density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
#print('Percentage Density of Non-Zero Values: ', density)

#split data into Train and Test Sets
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Train Model and Predict
from sklearn.naive_bayes import MultinomialNB
#model= MultinomialNB().fit(X_train,y_train)
#prediction = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
#cMatrix = confusion_matrix(y_test,prediction)
#sns.heatmap(cMatrix.T, square=True, annot=True, fmt='d', cbar=False)
#plt.xlabel('true Rating')
#plt.ylabel('predicted Rating');
#plt.show()

#print(classification_report(y_test,prediction))