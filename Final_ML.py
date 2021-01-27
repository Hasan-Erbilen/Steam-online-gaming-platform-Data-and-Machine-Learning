from types import ClassMethodDescriptorType
import json_lines
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

vote_txt = ["Not voted", "Voted" ]
access_txt = ["Full version", "Early Access" ]

#predict using knn 
def do_knn(x_train, y_train, test_data, k = 3):
    
    #create knn model using k(neighbour count) and cosine distance
    knn = KNeighborsClassifier(n_neighbors=k,metric=cosine_distances)
    
    #train the data
    knn.fit(x_train, y_train)
    
    #predict the output for test_data
    res = knn.predict(test_data)

    #return the result predicted
    return res

#predict using decision tree
def do_decision_tree(train_data, y_train, test_data):

    #create decision tree model
    dec_tree = DecisionTreeClassifier()

    #train the data
    dec_tree.fit(train_data, y_train)
    
    #predict the output for test_data
    res = dec_tree.predict(test_data)
    return res

#predict using logistic
def do_logistic(train_data, y_train, test_data):
    
    #create logistic regression model
    logistic = LogisticRegression(random_state=0)
    
    #train the data
    logistic.fit(train_data, y_train)
    
    #predict the output for test_data
    res = logistic.predict(test_data)
    return res

#predict using svm
def do_svm(train_data, y_train, test_data):
    
    #create svm model
    clf = svm.SVC()
    
    #train the data
    clf.fit(train_data, y_train)
    
    #predict the output for test_data
    res = clf.predict(test_data)
    return res

#data to read 

x = [] # array of review text 
y = [] # array of voted up state
z = [] # array of early access state

#read the data from file -------------
with open('reviews_200.jl', 'rb') as f:#open file
    #every line is json object
    for item in json_lines.reader(f):
        #append the data to variable
        x.append(item['text'])
        y.append(item['voted_up'] )
        z.append(item['early_access'] )

#create vectorizer to extract the features from the text
vectorizer = TfidfVectorizer( max_df=0.3)

#extract the features from the text
# so every review text is converted to number array
X = vectorizer.fit_transform(x)

#create sequence array of index from 0 to count of review
indices = np.arange(len(x))

#split the index array into train index array and test index array and the propotion is 0.001
#so the train count is 4995 if the count of review is 5000
#and the test  count is 5
train_id, test_id = train_test_split(indices, test_size=0.1)

#create train data from the train index array
train_x = X[train_id]
train_y = [y[idx] for idx in train_id]
train_z = [z[idx] for idx in train_id]

#select the first text for test
test_text = x[test_id[0]]

#extract the features from the test text
test_x = vectorizer.transform([test_text])
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
#output the test text
print('text:\n%.300s\n--------------'%test_text)

#do knn and print the result
print("KNN Regression : ")
test_y = do_knn(train_x, train_y, test_x)
test_z = do_knn(train_x, train_z, test_x)
scores = cross_val_score(KNeighborsClassifier(n_neighbors=3,metric=cosine_distances), X, y, cv=cv)
print("\t%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print( "\t", vote_txt[int(test_y)], access_txt[int(test_z)])

#do svm classifier and print the result
print( "SVM Regression : ")
test_y = do_svm(train_x, train_y, test_x)
test_z = do_svm(train_x, train_z, test_x)
scores = cross_val_score(svm.SVC(), X, y, cv=cv)
print("\t%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print( "\t", vote_txt[int(test_y)], access_txt[int(test_z)])

#do decision tree classifier and print the result
print( "Decision Tree result : ")
test_y = do_decision_tree(train_x, train_y, test_x)
test_z = do_decision_tree(train_x, train_z, test_x)
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=cv)
print("\t%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print( "\t", vote_txt[int(test_y)], access_txt[int(test_z)])

#do logistic regression and print the result
print( "Logistics Regression : ")
test_y = do_logistic(train_x, train_y, test_x)
test_z = do_logistic(train_x, train_z, test_x)
scores = cross_val_score(LogisticRegression(random_state=0), X, y, cv=cv)
print("\t%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print( "\t", vote_txt[int(test_y)], access_txt[int(test_z)])
