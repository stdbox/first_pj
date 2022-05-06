import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#Tạo danh sách từ dừng (stopword)
stopword=[]
stopword.append('.')
stopword.append(',')
stopword.append(';')
#stopword.append('ứng')
print(stopword)
df=pd.read_excel('D:/code_projects/code_projects/spam-titles-dev/spam-titles-dev/data/sentimentvn.xlsx')
#split into dependant and independent variable
X=df['text'].to_list()

#print('X=',x)

y=df['label'].to_list()
#print('y=',y)

#print(x)
#print(y)

#splitting dataset into a training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=10)

#print('X_train ban đầu= ',X_train)
#Vectorizer = TfidfVectorizer(analyzer = u'word',max_df=0.95,ngram_range=(1, 1),stop_words=set(stopword),max_features=4000)
#Convert a collection of text documents to a matrix of token counts.

#This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix.

Vectorizer = CountVectorizer(analyzer = u'word',ngram_range=(1, 1),stop_words=set(stopword))

word_count=Vectorizer.fit(X_train)
X_train=word_count.transform(X_train)
print(X_train.todense())
X_test=word_count.transform(X_test)

#Fitting Stochastic Gradient Descent (SGD) to training set (Huấn luyện mô hình Stochastic Gradient Descent (SGD))
from sklearn.linear_model import SGDClassifier
DC = SGDClassifier(loss="hinge", penalty="l2", max_iter=15)
DC.fit(X_train, y_train)

#predicting results (Dự đoán các đối tượng dữ liệu trong tập X_test)
y_pred = DC.predict(X_test)
print('Các độ đo Precision, Reacll, F1-Score đạt được từ mô hình phân lớp Stochastic Gradient Descent: \n',classification_report(y_test, y_pred, digits=4))
#print("accuracy score:", accuracy_score(y_test,y_pred))

print('confusion_matrix=',confusion_matrix(y_test, y_pred))
