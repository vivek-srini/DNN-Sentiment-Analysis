import numpy as np
import pandas as pd
reviews_train=[]          #a list to store all the reviews in your training set
reviews_test=[]           #a list to store all the reviews in your test set  
reviews=[]                #a list to store all the reviews including training and test sets
import glob
import os

#Write a function to read the text in every review
def text(filename):
    file=open(filename,'r')        
    text=file.read()
    file.close()
    return text


os.chdir(r'F:\sentiments_reviews\txt_sentoken\all reviews')
all_files=glob.glob('*.txt')
for file in all_files:
    reviews.append(text(file))                           #appends all reviews to the list


os.chdir(r'F:\sentiments_reviews\txt_sentoken\pos')
files_positive=glob.glob('*.txt')
for file in files_positive:
    reviews_train.append(text(file))                     #appends positive reviews to the training list


os.chdir(r'F:\sentiments_reviews\txt_sentoken\neg')
files_negative=glob.glob('*.txt')
for file in files_negative:
    reviews_train.append(text(file))                     #appends negative reviews to the training set


os.chdir(r'F:\sentiments_reviews\txt_sentoken\pos_test')
files_pos_test=glob.glob('*.txt')
for file in files_pos_test:
    reviews_test.append(text(file))                      #append positive reviews to test set
os.chdir(r'F:\sentiments_reviews\txt_sentoken\neg_test')
files_neg_test=glob.glob('*.txt')


for file in files_neg_test:
    reviews_test.append(text(file))                      #appends negative reviews to test set


y_1=np.ones((1,900))
y_2=np.zeros((1,900))
y_train=(np.append(y_1,y_2,axis=1).ravel())              #labels for training set
y_train=np.array(pd.get_dummies(y_train))


y_test=np.append(np.ones((1,100)),np.zeros((1,100)),axis=1).ravel()  #labels for test set
y_test=np.array(pd.get_dummies(y_test))


y_pos_neg=np.append(np.ones((1,1000)),np.zeros((1,1000)),axis=1).ravel()
y_pos_neg=np.array(pd.get_dummies(y_pos_neg))            #labels to create pos-neg ratios

vocab=set()
for i in range(len(reviews)):
    for word in reviews[i].split():          
        vocab.add(word)                                   #Create a vocabulary of every word in the reviews


num_features=len(vocab)
num_examples_train=len(reviews_train)
num_examples_test=len(reviews_test)


from collections import Counter
pos_count_train=Counter()                                #Keeping Counter of every word in the vocabulary
neg_count_train=Counter()
tot_count_train=Counter()
for i in range(len(reviews)):
    if np.argmax(y_pos_neg[i],axis=0)==1:
        for word in reviews[i].split():
            pos_count_train[word]+=1
            tot_count_train[word]+=1
    else:
        for word in reviews[i].split():
            neg_count_train[word]+=1
            tot_count_train[word]+=1


def pos_neg_ratio(word):
    return abs(np.log((pos_count_train[word])/float(neg_count_train[word]+1)))    #Function to calculate the positive-negative ratio of every word used to remove noise

 

def update_input_layer(reviews,num_examples):                 #Function to update the input Vector of the neural net
    rank_dict={}
    i=0
    for word in vocab:
        rank_dict[word]=i
        i+=1
    x=np.zeros((num_examples,num_features))
    for i in range(len(reviews)):
        for word in reviews[i].split():
            if word in vocab and pos_neg_ratio(word)>0.3:
                index=rank_dict[word]
                x[i][index]+=1
    return x

x_test=update_input_layer(reviews_test,num_examples_test)      #input test array
x_train=update_input_layer(reviews_train,num_examples_train)   #array passed as input to neural net

"""Next, we Train our Neural Net"""

from keras.models import Sequential
from keras.layers import Dense,Dropout
model=Sequential()
model.add(Dense(10,activation='relu',input_dim=x_train.shape[1]))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=10)
print("It's done!")


"""Now, we test our accuracy on a test set"""
score=model.evaluate(x_test,y_test)
print("Your test set accuracy is",score[1])

"""We get an accuracy of 82% on a test set of 200 reviews by training on just
   1000 positive and 1000 negative reviews"""


    