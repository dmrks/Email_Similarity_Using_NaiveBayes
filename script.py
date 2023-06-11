from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = fetch_20newsgroups()

#1
print(emails.target_names)

#2
train_emails =fetch_20newsgroups(categories =['comp.sys.ibm.pc.hardware','rec.sport.hockey'],subset='train',shuffle = True,random_state = 108)

#3
print(emails.data[5])

#4+5 = 1 -> Hockey
print(emails.target_names)

#6

test_emails = fetch_20newsgroups(categories =['comp.sys.ibm.pc.hardware','rec.sport.hockey'],subset='test',shuffle = True,random_state = 108)

#7+8
counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)

#9
train_counts = counter.transform(train_emails.data)

#10
test_counts = counter.transform(test_emails.data)

#11
classifier = MultinomialNB()

#12
classifier.fit(train_counts,train_emails.target)

#13 = 0.9723618090452262

print(classifier.score(test_counts,test_emails.target))

#14 = 0.9974715549936789

