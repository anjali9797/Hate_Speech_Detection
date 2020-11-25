# Hate_Speech_Detection
Hate Speech Detection is done on the data given to us 
# Major Tasks Done :

# 1. Removing Numeric Data:
If we observe the general pattern then numeric data doesnot contribute to
hate speech classification marginal data and can thus be ignored. We can
break the tweets into words and check whether the word was numeric or
not , if it was numeric then we need not consider it.
When I incorporated this approach I was able to see appropriate results in
the accuracy and hence this approach is included in my model.
# 2.Removing tagged users or username mentions:
All the usernames or data which was beginning with @ was removed as
mentioning the users does not have any relation with the task under
consideration.
# 3.Handling HashTags (#):
Various approaches were tried to handle the hash tag data.
First I tried removing # and then separating the owrds using wordsegment
but didn’t observe any particular improvement in my prediction model.
As each preprocessing comes with a cost and if a particular preprocessing
is not bearing expected results we can ignore it.
Second I tried removing just the ‘#’ symbol abd then the entire text
associated with # , the latter flared well and was perhaps incorporated in
the final model.
# 4.Removing website mentions i.e. links of the type https:
Links to websites itself do not imply any specific criteria which could help
us and also increases the processing time. So we donot include any such
data.
# 5.Removing stopwords:
Stopwords are words which are general words and do not convey any
special meaning . So we remove this data from the main data .
# 6.Removing punctutation marks and other signs:
The punctuation marks do not add any specific value and add the same
meaning whether used in hate speech or in normal speech so these can be
ignored.
# 7.Removing words smaller than length 2:
These words do not convey a specific meaning and it was observed that
these can be ignored.
# 8. Emoticons:
Handling emoticons is a herculean task as it was not possible for a single
demoji to decode all the emoticons and then translate them to words and
use. Tried it using some words but didnt work out well. Also dropping out
emoticons didnt flare out well. So emoticons were left as it is.
# 9.Using Only Dictionary words:
Using only the dictionary words did not have a huge impact on the
predictions but were increasing the processing time considerably so this
approach was dropped.

# Training the Model:
The model was trained using SVM .The data was preprocessed using Tfid
Vectorizer.
