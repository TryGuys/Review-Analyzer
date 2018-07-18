import pandas as pd
import re
from nltk.stem import porter
import pickle

filename="Review_data_cleaned.tsv"
df=pd.read_csv(filename,delimiter='\t')

X=df['Review']
y=df['labels']

#Data Preprocessing
#removing symbols and tokenizing
data=[re.findall('[a-zA-Z]+',_) for _ in X]
#stemming data 
ps=porter.PorterStemmer()
for _ in range(len(data)):
    data[_]=" ".join([ps.stem(token) for token in data[_]])
    
#splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data,y,test_size=0.2,random_state=213)

from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer(stop_words='english')
X_train=tv.fit_transform(X_train)

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(X_train,y_train)

#Saving model
f=open("vectorizer.pickle",'wb')
pickle.dump(tv,f)
f.close()


import numpy as np
test=np.array(["This is interesting very much"])
test=tv.transform(test)
res=nb.predict(test)
print(res[0])