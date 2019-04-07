# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:56:00 2019

@author: Brandon Townsend
"""

import numpy as np
import pandas as pd
import re
import time
t_e = [] # traning examples
T = 20 # max number of training iterations
w = [] # weight vector (will be same length as feature vector)
y_hat = 0.0 #prediction
n = 1 # learning rate
start_time = time.time()
#Pre-processing
path = 'C:/Users/Brandon Townsend/Downloads/Shellscriptexample3/11426975/data'
traind = open(path+'/traindata.txt','r').read()
trainl = open(path+'/trainlabels.txt','r').read()
testd = open(path+'/testdata.txt','r').read()
testl = open(path+'/testlabels.txt','r').read()
stopw = open(path+'/stoplist.txt','r').read()

traindata = re.split('[ \n]',traind)
trainlabels = re.split('[ \n]',trainl)

testdata = re.split('[ \n]',testd)
testlabels = re.split('[ \n]',testl)

vocab = np.unique(traindata).tolist()
del vocab[0]

newvocab = []
for word in range(len(vocab)):
    match = re.search(vocab[word], stopw)
    if match == None:
        newvocab.append(vocab[word])
        
vocab = newvocab
del newvocab

vocabt = np.unique(testdata).tolist()
del vocabt[0]

newvocab = []
for word in range(len(vocab)):
    match = re.search(vocab[word], stopw)
    if match == None:
        newvocab.append(vocab[word])
        
vocabt = newvocab
del newvocab

messagesp = re.split('\n',traind)
messages={}

messagespt = re.split('\n',testd)
messagest={}

for i in range(len(messagesp)):
    if trainlabels[i]!='':
        messages[i] = (re.split(' ',messagesp[i]), int(trainlabels[i]))

    
data = {}
temp = {}

for i in range(len(messages)):
    for j in messages[i][0]:

        if j in vocab:
            temp[j]= 1
            
    data[i]=dict(temp)
    temp.clear()
    
for i in range(len(messagespt)):
    if testlabels[i]!='':
        messagest[i] = (re.split(' ',messagespt[i]), int(testlabels[i]))

    
datat = {}
tempt = {}

for i in range(len(messagest)):
    for j in messagest[i][0]:
        if j in vocab:
            temp[j]= 1
            
    datat[i]=dict(temp)
    tempt.clear()
    
ftVectors = pd.SparseDataFrame(data, index = vocab, columns = range(len(messages)) )
ftVectorst = pd.SparseDataFrame(data, index = vocab, columns = range(len(messages)) )
#Classification

# Algorithm 1


file = open('./output.txt','w') 
file.write("Algorithm 1\n")
w = np.zeros([1,len(vocab)])
w_avg_cached = np.zeros([1,len(vocab)])
mstks = 0
mstkst = 0
mstksAvg = 0
y_hat = 0
corrections = 0
c = 1;
x_axis = range(len(messages))
indeces = range(len(messages))
indeces_t = range(len(messagest))
#totalMistakes = []
for j in range(T):
    mistakes = np.zeros([1, len(messages)])
    mstks = 0
    mstkst = 0
    mstksAvg = 0
    index = 0
    indext = 0
    wold = w
    wn = 1

    for i in indeces:
        y_hat = np.sign(np.dot(w,np.asmatrix(ftVectors[[i]].fillna(0).to_numpy())))

        if y_hat != messages[i][1]:
            mstks+=1
            corrections+=1
            f = n*messages[i][1]*np.array(ftVectors[i].fillna(0))
            w = w + f
            w_avg_cached = w_avg_cached + c*f
        index+=1
        c+=1
        wn+=1
        
    #averaged weights: w-1/c*u      
    if j==T-1:
        w_avg = w - 1/float(wn)*w_avg_cached

    
    for i in indeces_t:
        y_hat = np.sign(np.dot(w,np.asmatrix(ftVectors[[i]].fillna(0).to_numpy())))
#        if y_hat == float('nan'):
#            y_hat = 1
        if j==T-1:
            y_hat_avg = np.sign(np.dot(w_avg,np.asmatrix(ftVectors[[i]].fillna(0).to_numpy())))
                
            if y_hat_avg !=  messagest[i][1]:
                mstksAvg+=1
            

        if y_hat !=  messagest[i][1]:
             mstkst+=1
        
        indext+=1
        
    
    file.write("iteration-" + str(j+1) + " Mistakes " + str(mstkst) + "\n")
    file.write("iteration-" + str(j+1) + " training-accuracy " + str((100*(1.0-float(mstks)/index))) + "% testing-accuracy " + str((100*(1.0-float(mstkst)/indext))) + '%\n')

file.write("training-accuracy-standard-perceptron " + str((100*(1.0-float(mstks)/index))) + "% testing-accuracy-averaged-perceptron " + str((100*(1.0-float( mstksAvg)/indext))) + '%\n')

#    mp.plot(x_axis, mistakes.T.tolist(), label = 'Pass ' + str(j))
#    mp.legend(loc='upper left')
#    mp.show()
#    totalMistakes.append(mstks)
    
#mp.plot(range(len(totalMistakes)), totalMistakes)
#mp.show()
file.close() 


t_e = [] # traning examples
T = 20 # max number of training iterations

y_hat = 0.0 #prediction
n = 1 # learning rate

#Pre-processing

traind = open(path+'/ocr_train.txt','r').read()
testd = open(path+'/ocr_test.txt','r').read()


traindtemp = re.split('\d+\tim|\t_\n|\t+\n',traind)
testtemp = re.split('\d+\tim|\t_\n|\t+\n',testd)
traindata = filter(None, traindtemp)
testdata = filter(None, testtemp)

binary = []
labels = []
binaryt = []
labelst = []

uniq = []
for pair in traindata:
    t = re.split('\t', pair)
    t2 = [float(i) for i in list(t[0])]
    uniq.append(t[1])
    binary.append(t2)
    labels.append(t[1])
    
uniq = np.unique(uniq)

for pair in testdata:
    t = re.split('\t', pair)
    t2 = [float(i) for i in list(t[0])]
    binaryt.append(t2)
    labelst.append(t[1])
    


trainpairs = pd.DataFrame(columns = ['binary','labels'])
trainpairs['binary']=binary
trainpairs['labels']=labels

testpairs = pd.DataFrame(columns = ['binary','labels'])
testpairs['binary']=binaryt
testpairs['labels']=labelst

ftVectors = pd.DataFrame(index = range(128), columns = uniq).fillna(0)
avgPweights_cached = pd.DataFrame(index = range(128), columns = uniq).fillna(0)

# break ties when multiple weights have same score

##Classification
#
## Algorithm 2


w = np.zeros([1,len(uniq)]) # weight vector (will be same length as feature vector)
w_t_hat = np.zeros([1,len(uniq)])
y_hat = 0
#yy_t_hat = '0'
corrections = 0
correctionst = 0
x_axis = range(len(trainpairs))
indeces = range(len(trainpairs))
totalMistakes = []
totalMistakest = []

mstks = 0
mstkst = 0
mstksAvg = 0
index = 0
indext=0
file = open('./output.txt','a')
file.write("\nAlgorithm 2\n")

exCounter = 1

for j in range(T):

    mistakes = np.zeros([1, len(trainpairs)])
    mstks = 0
    mstkst = 0
    mstksAvg = 0
    index = 0
    wn = 1
#    shuffle(indeces)
    for i in trainpairs.index:
        preval = 0.0
        y_t_hat = '0'
        
        dprods = ftVectors.T.dot(trainpairs['binary'][i])
        y_t_hat = np.argmax(dprods)

        if y_t_hat != trainpairs['labels'][i]:
            mstks+=1
            corrections+=1
            
            ftVectors[trainpairs['labels'][i]] = [wt + lr*x for wt,x,lr in zip (ftVectors[trainpairs['labels'][i]] , trainpairs['binary'][i], [n]*128)]

            ftVectors[y_t_hat] = [wt - lr*x for wt,x,lr in zip (ftVectors[y_t_hat] , trainpairs['binary'][i], [n]*128)]
            
            # For averaged perceptron: updating cahced weights
            avgPweights_cached[trainpairs['labels'][i]] = [wt + lr*exC*x for wt,x,lr,exC in zip (avgPweights_cached[trainpairs['labels'][i]] , trainpairs['binary'][i], [n]*128, [exCounter]*128)]
            exCounter = 1
            wn+=1
            
        exCounter+=1    
        index+=1
    
    #averaged weights: w-1/c*u    
        if j==T-1:
            avgWeights = ftVectors - 1/float(wn)*avgPweights_cached
    indext=0
    
    for i in testpairs.index:
        preval = 0.0
        y_t_hat = '0'
        
        dprods = ftVectors.T.dot(testpairs['binary'][i])
        y_t_hat = np.argmax(dprods)
        
        if j==T-1:
            y_t_hatAvg = '0'
            dprodsAvg = avgWeights.T.dot(testpairs['binary'][i])
            y_t_hatAvg = np.argmax(dprodsAvg)
            if y_t_hatAvg != testpairs['labels'][i]:
                mstksAvg+=1

        if y_t_hat != testpairs['labels'][i]:
            mstkst+=1
        
        indext+=1
        
    file.write("iteration-" + str(j+1) + " Mistakes " + str(mstkst) + "\n")
    file.write("iteration-" + str(j+1) + " training-accuracy " + str((100*(1.0-float(mstks)/index))) + "% testing-accuracy " + str((100*(1.0-float(mstkst)/indext))) + '%\n')

file.write("training-accuracy-standard-perceptron " + str((100*(1.0-float(mstks)/index))) + "% testing-accuracy-averaged-perceptron " + str((100*(1.0-float( mstksAvg)/indext))) + '%')
end_time = time.time()

 

Run_time=end_time-start_time
#print Run_time
file.close()