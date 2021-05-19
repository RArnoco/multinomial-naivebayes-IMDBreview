#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score


# In[33]:


df=pd.read_csv("datasets/IMDB Dataset.csv" ,  names=['review','sentiment'])


# In[34]:


df.head()


# In[35]:


#TFIDF Vectorizer
stopset= set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)


# In[36]:


#dependent variable
y= df.sentiment


# In[37]:


#convert df.txt from text to features
X= vectorizer.fit_transform(df.review)


# In[38]:


#WORDS
print (y.shape)
print (X.shape)


# In[39]:


#test train up to 20%
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)


# In[40]:


#train naive bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)


# In[41]:


#accuracy
roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])


# In[45]:


#output
movie_reviews_array=np.array(["ugly"])
movie_review_vector = vectorizer.transform(movie_reviews_array)
print (clf.predict(movie_review_vector))


# In[ ]:




