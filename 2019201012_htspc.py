#!/usr/bin/env python
# coding: utf-8

# In[127]:


import pandas as pd
import csv
import numpy as np
import matplotlib
import random
from sklearn.metrics import accuracy_score
import os
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
import re
import nltk
from nltk.corpus import stopwords
from wordsegment import load, segment


# In[128]:


df=pd.read_csv("train.csv")


# # Data Preprocessing

# Separating the data and the labels

# In[129]:


text_df=df.get("text")

label_df=df.get("labels")
stops=set(stopwords.words('english'))
#changes
from nltk.corpus import words
setofwords = set(words.words())


# In[269]:


datas=text_df.values.tolist()

for i in range(len(datas)):
    s=datas[i]
   
    
    #converting to lower case for convinience
    s=s.lower().split()
    #now let us remove the stop words
    tes=list()
    for w in s:
        if "https" in w:
            continue
        if w not in stops:
            if w.isnumeric()==False:
                if w[0]!='@' and w[0]!="#":
                    if len(w)>=2:
                        tes.append(w)
    tes=" ".join(tes)
   
    #replacing nums and other characters
    tes=re.sub(r'https?://[A-Za-z0-9./]+','',tes)
    tes=re.sub(r"[0-9^,!.\/''+-=]"," ",tes)
    tes=re.sub(r","," ",tes)
    tes=re.sub(r"!"," ",tes)
    tes=re.sub(r"#","",tes)
    tes=re.sub(r"$","",tes)
    tes=re.sub(r"%","",tes)
    tes=re.sub(r"^","",tes)
    tes=re.sub(r"&","",tes)
    tes=re.sub(r"\*","",tes)
    tes=re.sub(r"\("," ",tes)
    tes=re.sub(r"\)"," ",tes)
    tes=re.sub(r"_"," ",tes)
    tes=re.sub(r":","",tes)
    tes=re.sub(r"-"," ",tes)
    
    
    """emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    #tes=emoji_pattern.sub(r'', tes)"""
    datas[i]=tes
    
#data is cleaned
    
    
    
    


# In[270]:


#data is clean
print(len(datas))
test_data=list()
actual_labels=list()
tester=pd.read_csv("testq1.csv")
test_df=tester.get("text")
test_data=test_df.values.tolist()
for i in range(len(test_data)):
    s=test_data[i]
    
    #converting to lower case for convinience
    s=s.lower().split()
    #now let us remove the stop words
    tes=list()
    for w in s:
        if "https" in w:
            continue
        if w not in stops:
            if w.isnumeric()==False:
                if w[0]!='@' and w[0]!="#":
                    tes.append(w)
                    if len(w)>=2:
                        tes.append(w)
    tes=" ".join(tes)
   
    #replacing nums and other characters
    tes=re.sub(r'https?://[A-Za-z0-9./]+','',tes)
    tes=re.sub(r"[0-9^,!.\/''+-=]"," ",tes)
    tes=re.sub(r","," ",tes)
    tes=re.sub(r"!"," ",tes)
    #tes=re.sub(r"#","",tes)
    tes=re.sub(r"$","",tes)
    tes=re.sub(r"%","",tes)
    tes=re.sub(r"^","",tes)
    tes=re.sub(r"&","",tes)
    tes=re.sub(r"\*","",tes)
    tes=re.sub(r"\("," ",tes)
    tes=re.sub(r"\)"," ",tes)
    tes=re.sub(r"_"," ",tes)
    tes=re.sub(r":","",tes)
    tes=re.sub(r"-"," ",tes)
    
    
   
    #removing emojis from the data
    """emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    #tes=emoji_pattern.sub(r'', tes)"""
    test_data[i]=tes
    


# In[271]:


#splitting in train and test
labs=label_df.values.tolist()
r=random.randint(1,10)
ls=list()
lab=list()
"""
test_data=list()
actual_labels=list()
#tester=pd.read_csv("test.csv")
#test_df=tester.get("text")
#test_data=test_df.values.tolist()
test_data=list()

for x in range(len(datas)-1):
    if np.random.random()<0.9:
        ls.append(datas[x])
        lab.append(labs[x])
    else:
        test_data.append(datas[x])
        actual_labels.append(labs[x])
"""
for x in datas:
    ls.append(x)
for x in labs:
    lab.append(x)
#vectorizing using tfvectorizer"""

vect = TfidfVectorizer(min_df=0.000001,stop_words='english').fit(ls)

x_train=vect.transform(ls)


#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()
#model.fit(x_train, lab)

from sklearn import svm
clf = svm.SVC()
clf.fit(x_train, lab)
predictions=clf.predict(vect.transform(test_data))
#prediction=model.predict(vect.transform(test_data))



# In[ ]:





# In[264]:




# In[276]:


#writing in file
f1=open("anshate.csv","w")
f1.write(",labels")
c=0
for i in predictions:
    st=str(i)
    f1.write("\n")
    f1.write(str(c))
    c=c+1
    f1.write(",")
    f1.write(st)
    
f1.close()
    


# In[273]:


  


# In[ ]:





# In[ ]:




