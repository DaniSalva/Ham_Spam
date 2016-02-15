# Javier Civera, jcivera@unizar.es
# December 2015
# Spam filtering using Naive Bayes

######################################################
# Imports
######################################################

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import json
import glob
import os
import sys
from sklearn import metrics
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.metrics import classification_report



import random
from blaze import inf
from sklearn.metrics.classification import accuracy_score
from types import ClassType
from Crypto.Random.random import shuffle
from sklearn.svm.libsvm import predict, cross_validation
from sklearn.cross_validation import cross_val_score


######################################################
# Aux. functions
######################################################

# load_enron_folder: load training, validation and test sets from an enron path
def load_enron_folder(path):

   ### Load ham mails ###
   # List mails in folder
   ham_folder = path + '\ham\*.txt'
   ham_list = glob.glob(ham_folder)
   num_ham_mails = len(ham_list)

   ham_mail = []
   for i in range(0,num_ham_mails):
      ham_i_path = ham_list[i]
      '''print(ham_i_path)'''
      # Open file
      ham_i_file = open(ham_i_path, 'r')
      # Read
      ham_i_str = ham_i_file.read()
      # Convert to Unicode
      ham_i_text = ham_i_str.decode('utf-8',errors='ignore')
      # Append to the mail structure
      ham_mail.append(ham_i_text)
      # Close file
      ham_i_file.close()
   
   random.shuffle(ham_mail)

   # Load spam mails

   spam_folder = path + '\spam\*.txt'
   spam_list = glob.glob(spam_folder)
   num_spam_mails = len(spam_list)

   spam_mail = []
   for i in range(0,num_spam_mails):
      spam_i_path = spam_list[i]
      '''print(spam_i_path)'''
      # Open file
      spam_i_file = open(spam_i_path, 'r')
      # Read
      spam_i_str = spam_i_file.read()
      # Convert to Unicode
      spam_i_text = spam_i_str.decode('utf-8',errors='ignore')
      # Append to the mail structure
      spam_mail.append(spam_i_text)
      # Close file
      spam_i_file.close()

   random.shuffle(spam_mail)

   # Separate into training, validation and test
   num_ham_training = int(round(0.8*num_ham_mails))
   ham_training_mail = ham_mail[0:num_ham_training]
   '''print(num_ham_mails)
   print(num_ham_training)
   print(len(ham_training_mail))'''
   ham_training_labels = [0]*num_ham_training
   '''print(len(ham_training_labels))'''
    
   num_ham_validation = int(round(0.1*num_ham_mails))
   ham_validation_mail = ham_mail[num_ham_training:num_ham_training+num_ham_validation]
   '''print(num_ham_validation)
   print(len(ham_validation_mail))'''
   ham_validation_labels = [0] * num_ham_validation
   '''print(len(ham_validation_labels))'''

   ham_test_mail = ham_mail[num_ham_training+num_ham_validation:num_ham_mails]
   '''print(num_ham_mails-num_ham_training-num_ham_validation)
   print(len(ham_test_mail))'''
   ham_test_labels = [0] * (num_ham_mails-num_ham_training-num_ham_validation)
   '''print(len(ham_test_labels))'''

   num_spam_training = int(round(0.8*num_spam_mails))
   spam_training_mail = spam_mail[0:num_spam_training]
   '''print(num_spam_mails)
   print(num_spam_training)
   print(len(spam_training_mail))'''
   spam_training_labels = [1]*num_spam_training
   '''print(len(spam_training_labels))'''

   num_spam_validation = int(round(0.1*num_spam_mails))
   spam_validation_mail = spam_mail[num_spam_training:num_spam_training+num_spam_validation]
   '''print(num_spam_validation)
   print(len(spam_validation_mail))'''
   spam_validation_labels = [1] * num_spam_validation
   '''print(len(spam_validation_labels))'''

   spam_test_mail = spam_mail[num_spam_training+num_spam_validation:num_spam_mails]
   '''print(num_spam_mails-num_spam_training-num_spam_validation)
   print(len(spam_test_mail))'''
   spam_test_labels = [1] * (num_spam_mails-num_spam_training-num_spam_validation)
   '''print(len(spam_test_labels))'''

   training_mails = ham_training_mail + spam_training_mail
   training_labels = ham_training_labels + spam_training_labels
   validation_mails = ham_validation_mail + spam_validation_mail
   validation_labels = ham_validation_labels + spam_validation_labels
   test_mails = ham_test_mail + spam_test_mail
   test_labels = ham_test_labels + spam_test_labels

   data = {'training_mails': training_mails, 'training_labels': training_labels, 'validation_mails': validation_mails, 'validation_labels': validation_labels, 'test_mails': test_mails, 'test_labels': test_labels} 

   return data

#class_type: type of classifier
#k: number of folds
def kfold_cross_validation(class_type,k,dataMails,dataLabels):
    '''
    K-fold cross validation: we split the data set into k parts, hold out one,
    combine the others and train on them, then validate against the held-out 
    portion. We repeat that process k times (each fold), holding out a different 
    portion each time. Then we average the score measured for each fold 
    to get a more accurate estimation of the model's performance.
    '''
    bestSize=0
    bestScore=0.0
    bestAccuracy=0.0
    label=np.array(dataLabels)
    print "\nKFOLD"
    print "*******"
    alp=0;
    for alp in range(0, 10):
        scores=0.0
        accuracy=0.0
        k_fold = KFold(n=len(dataLabels), n_folds=k,shuffle=True,random_state=None)
        confusion = np.array([[0, 0], [0, 0]])

        for train_indices, test_indices in k_fold:     
          
            data_train = dataMails[train_indices]
            labels_train = label[train_indices]
            
            data_test = dataMails[test_indices]
            labels_test = label[test_indices]
            
            if (class_type=='Bernouilli'):
                classifier = BernoulliNB(alpha=alp, fit_prior=True, class_prior=None)
            else:
                classifier = MultinomialNB(alpha=alp, fit_prior=True, class_prior=None)
                
            classifier.fit(data_train,labels_train)            
            predictions = classifier.predict(data_test)
            
            scores += f1_score(labels_test, predictions)
            accuracy+=accuracy_score(labels_test,predictions)
                  
            confusion += confusion_matrix(labels_test, predictions)
                  
        accuracyMean= accuracy/k            
        if (accuracyMean> bestAccuracy):
            bestAccuracy=accuracy/k
            
        scoresMean= scores/k;
        print 'Laplace value: %d' % alp
        print 'ScoreMean: '+ str(scoresMean)
        print 'AccuracyMean:'+ str(accuracyMean)
        
        if (scoresMean> bestScore):
            bestScore=scoresMean
            bestSize=alp                       
          
    print "\nKFOLD Result"
    print "*************"
    print 'BestAccuracy: '+ str(bestAccuracy)
    print 'BestScore: '+ str(bestScore)
    print 'BestSize: '+ str(bestSize)
    print "\n"
        
    return bestSize

#stopW: boolean variable to decide whether the model uses stop_words or not
def BagWords(stopW):
    if (stopW):
        return CountVectorizer(stop_words='english')
    else:
        return CountVectorizer()

#stopW: boolean variable to decide whether the model uses stop_words or not
def Bigrams(stopW):
    if (stopW):
        return CountVectorizer(ngram_range=(2,2),stop_words='english')
    else:
        return CountVectorizer(ngram_range=(2,2))

def Bernouilli(CV,dataMails,dataTest):
    CV.binary=True
    train_data=CV.fit_transform(dataMails)  
    test_data=CV.transform(dataTest) 
    return (train_data,test_data)
    
# CV: Bag Words model already built
# dataMails
# dataTest
# binary: boolean variable to decide whether the model is boolean or not
# freq: boolean variable to decide whether the model is normalized or not

def Multinomial(CV,dataMails,dataTest,binary,freq):
    if(binary):
        CV.binary=True;
    train_data_bag=CV.fit_transform(dataMails)
    '''learns the vocabulary (fit) and extracts word count features (transform)'''  
    test_data_bag=CV.transform(dataTest)
    
    if(binary==False and freq==True):
        transformer=TfidfTransformer()
        train_data_freq=transformer.fit_transform(train_data_bag) 
        '''inicialize + transform our count-matrix to a tf-idf representation'''
        test_data_freq=transformer.transform(test_data_bag) 
        return (train_data_freq, test_data_freq)
    else:
        return (train_data_bag,test_data_bag)
    
#typ: type of model -> Bigrams/Boolean... (String)
#bestAlpha: best Laplace smoothing value (int)
#clas_type: type of classifier (String)
def trainClassifier(train_data_freq,dataLabels,typ,bestAlpha,clas_type):
    
    
    if (bestAlpha!=-1):
        bestParam=bestAlpha
    else:
        k_folds=9
        bestParam=kfold_cross_validation(clas_type,k_folds, train_data_freq, dataLabels)
        
    print "***********************"
    print clas_type+" NAIVE BAYES"
    print "***********************"
    print "------ "+typ+" ------\n"
    
    print ('Train the classifier with best alpha value (Laplace smoothing)')
    
    if(clas_type=='Multinomial'):
        classifier = MultinomialNB(alpha=bestParam, fit_prior=True, class_prior=None)
    else:
        classifier = BernoulliNB(alpha=bestParam, fit_prior=True, class_prior=None)    
        
    classifier.fit(train_data_freq,dataLabels)
    
    predictions=classifier.predict(train_data_freq)
    
    score = f1_score(dataLabels, predictions)
    print "Alpha: %d" % bestParam
    print "Score: " + str(score)
        
    return (classifier,bestParam)

def getPredictions(test_data,classifier,cv):
    predictions=classifier.predict(test_data)
    return predictions

def evaluation(predictions,predText,test_mails,test_labels):

    precision = dict()
    recall = dict()
    max_f1=0;
    best_clas=0;
    for i in range(0,len(predictions)):
        score=f1_score(test_labels,predictions[i])
        accuracy = accuracy_score(test_labels,predictions[i])
        confusion=confusion_matrix(test_labels, predictions[i])
        precision[i], recall[i], _ =precision_recall_curve(test_labels,predictions[i])
        if (score>max_f1):
            max_f1=score
            best_clas=i
        
        print "\n"
        print '**TEST '+predText[i]+' **'
        print 'accuracy: '+ str(accuracy)
        print 'f1score: '+ str(score)
        print 'error: '+ str((1-score))
        print confusion
            
    # Plot Precision-Recall curve
    plt.clf()
    for i in range(0,len(predictions)):
        plt.plot(recall[i], precision[i],
                 label='Prec-rec of '+predText[i]) 
        
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall SPAM detector')
    plt.legend(loc="lower left")
    plt.show()
    
    print '\n====>>> Best Classifier ='+predText[best_clas]
    predictions=predictions[best_clas]
    score=f1_score(test_labels,predictions)
    confusion=confusion_matrix(test_labels, predictions)
    print(classification_report(test_labels, predictions, target_names=['0','1']))
    print '\nError:'+str(1-score)
    print 'Confusion matrix:'
    print confusion
    
    falseHam=[] #Clasificados como Ham pero son Spam
    falseSpam=[]
    ham=[]
    spam=[]
    i=0
    for i in range (0,(len(test_labels))):
        if (test_labels[i]==1 and predictions[i]==0):
            falseHam.append(test_mails[i])
        if (test_labels[i]==0 and predictions[i]==1):
            falseSpam.append(test_mails[i])
        if (test_labels[i]==0 and predictions[i]==0):
            ham.append(test_mails[i])
        if (test_labels[i]==1 and predictions[i]==1):
            spam.append(test_mails[i])
        
    
    print "\n*********************"
    print "Examples of emails"
    print "*********************"
    print "\nFALSE HAMS (Classified as HAM when it was SPAM): "+str(len(falseHam))
    print "*************************************************"

    for i in range (0,2):
        print "\n\t"+falseHam[i]
    
    print "\nFALSE SPAMS (Classified as SPAM when it was HAM): "+str(len(falseSpam)) 
    print "*************************************************"
    

    for i in range (0,2):
        print "\n\t"+falseSpam[i]
    
    print "\nSPAMS: "+str(len(spam)) 
    print "********************"
    

    for i in range (0,2):
        print "\n\t"+spam[i]
        
    print "\nHAMS: "+str(len(ham))  
    print "********************"
   

    for i in range (0,2):
        print "\n\t"+ham[i]
                         
    return 0

######################################################
# Main
######################################################

print("Starting...")

print('Loading files...')
args=sys.argv[1:]
fold=args[0]
print fold
data=[]
listing = os.listdir(fold)
for infile in listing:
    print "Loading..." + infile
    dat=load_enron_folder(fold+infile)
    data.append(dat)
     
print('Files imported succesfully')

debug=args[1]

training_mails =   data[0]['training_mails']+data[1]['training_mails']+data[2]['training_mails']+data[3]['training_mails']+data[4]['training_mails']+data[5]['training_mails']
training_labels =  data[0]['training_labels']+data[1]['training_labels']+data[2]['training_labels']+data[3]['training_labels']+data[4]['training_labels']+data[5]['training_labels']
validation_mails = data[0]['validation_mails']+data[1]['validation_mails']+data[2]['validation_mails']+data[3]['validation_mails']+data[4]['validation_mails']+data[5]['validation_mails']
validation_labels =data[0]['validation_labels']+data[1]['validation_labels']+data[2]['validation_labels']+data[3]['validation_labels']+data[4]['validation_labels']+data[5]['validation_labels']

trainval_mails= training_mails+validation_mails;
trainval_labels=training_labels+validation_labels

test_mails = data[0]['test_mails']+data[1]['test_mails']+data[2]['test_mails']+data[3]['test_mails']+data[4]['test_mails']+data[5]['test_mails']
test_labels = data[0]['test_labels']+data[1]['test_labels']+data[2]['test_labels']+data[3]['test_labels']+data[4]['test_labels']+data[5]['test_labels']

ClasifTypes =     {0 : Multinomial,
                   1 : Bernouilli,
}
ModelTypes =     {0 : BagWords,
                  1 : Bigrams,
}

preds=[]
predsText=[]

if (debug==0):
    '''Multinomial (binary) - Bigrams with StopWords'''
    model=ModelTypes[1](True)
    dataMails_train,dataMails_test=ClasifTypes[0](model,trainval_mails,test_mails,True,False)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"Bigrams",1,'Multinomial')
    predictions6=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions6)
    predsText.append('Multinomial(binary) - Bigrams_StopWords')
    
    '''Multinomial (binary) - Bigrams without StopWords'''
    model=ModelTypes[1](False)
    dataMails_train,dataMails_test=ClasifTypes[0](model,trainval_mails,test_mails,True,False)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"Bigrams",bestParam,'Multinomial')
    predictions6=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions6)
    predsText.append('Multinomial(binary) - Bigrams')
    
    print "\n"
    print "*******************"
    print "****EVALUATION*****"
    print "*******************"
    evaluation(preds,predsText, test_mails, test_labels)
    
elif(debug==1):
    '''Multinomial - Bigrams with frequencies'''
    model=ModelTypes[1](True)
    dataMails_train, dataMails_test=ClasifTypes[0](model,trainval_mails,test_mails,False,True)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"Bigrams",1,'Multinomial')
    predictions1=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions1)
    predsText.append('Multinomial - Bigrams_Freq')
    
    '''Multinomial - Bigrams without frequencies'''
    model=ModelTypes[1](True)
    dataMails_train, dataMails_test=ClasifTypes[0](model,trainval_mails,test_mails,False,False)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"Bigrams",bestParam,'Multinomial')
    predictions1=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions1)
    predsText.append('Multinomial - Bigrams')
    
    print "\n"
    print "*******************"
    print "****EVALUATION*****"
    print "*******************"
    evaluation(preds,predsText, test_mails, test_labels)
else:
    stopWords=True 
    normalizeBag=False
    
    '''Multinomial - Bigrams'''
    model=ModelTypes[1](stopWords)
    dataMails_train, dataMails_test=ClasifTypes[0](model,trainval_mails,test_mails,False,normalizeBag)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"Bigrams",-1,'Multinomial')
    predictions1=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions1)
    predsText.append('Multinomial - Bigrams')
    
    '''Multinomial - BagWords'''
    model=ModelTypes[0](stopWords)
    dataMails_train,dataMails_test=ClasifTypes[0](model,trainval_mails,test_mails,False,normalizeBag)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"BagWords",bestParam,'Multinomial')
    predictions2=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions2)
    predsText.append('Multinomial - BagWords')
    
    
    '''Bernouilli - Bigrams'''
    model=ModelTypes[1](stopWords)
    dataMails_train,dataMails_test=ClasifTypes[1](model,trainval_mails,test_mails)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"Bigrams",bestParam,'Bernouilli')
    predictions3=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions3)
    predsText.append('Bernouilli - Bigrams')
    
    
    '''Bernouilli - BagWords'''
    model=ModelTypes[0](stopWords)
    dataMails_train,dataMails_test=ClasifTypes[1](model,trainval_mails,test_mails)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"BagWords",bestParam,'Bernouilli')
    predictions4=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions4)
    predsText.append('Bernouilli - BagWords')
    
    
    '''Multinomial (binary) - BagWords'''
    model=ModelTypes[0](stopWords)
    dataMails_train,dataMails_test=ClasifTypes[0](model,trainval_mails,test_mails,True,normalizeBag)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"BagWords",bestParam,'Multinomial')
    predictions5=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions5)
    predsText.append('Multinomial (binary) - BagWords')
    
    
    '''Multinomial (binary) - Bigrams'''
    model=ModelTypes[1](stopWords)
    dataMails_train,dataMails_test=ClasifTypes[0](model,trainval_mails,test_mails,True,normalizeBag)
    classifier,bestParam=trainClassifier(dataMails_train,trainval_labels,"Bigrams",bestParam,'Multinomial')
    predictions6=getPredictions(dataMails_test,classifier,model)
    preds.append(predictions6)
    predsText.append('Multinomial (binary) - Bigrams')
    
    print "\n"
    print "*******************"
    print "****EVALUATION*****"
    print "*******************"
    evaluation(preds,predsText, test_mails, test_labels)

    


