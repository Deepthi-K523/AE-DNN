


import numpy as np
import os
from matplotlib import pyplot
from numpy import interp
#import matplotlib.pyplot as plt
import sklearn, tensorflow
import xlsxwriter 
import xlrd
from sklearn import svm #, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import model_selection

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io

from keras.layers import Input, Dense
from keras.engine.training import Model
from keras.models import Sequential, model_from_config,Model
from keras.layers.core import  Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.constraints import maxnorm



def prepare_data(seperate=False):
    print ("loading data")
    
    
    
    disease_fea = np.loadtxt("integrated-disease similarity.txt",dtype=float,delimiter=",")
    circRNA_fea  = np.loadtxt("integrated-cRNA similarity.txt",dtype=float,delimiter=",")
    interaction = np.loadtxt("interaction.txt",dtype=int,delimiter=",")
   
    
   
                        
    
    link_number = 0
    #nonlink_number=0
    train = []  
    testfnl= []       
    label1 = []
    label2 = []
    label22=[]
    ttfnl=[]
    #link_position = []
    #nonLinksPosition = []
    
    for i in range(0, interaction.shape[0]):   # shape[0]  returns no. of rows of matrix
        for j in range(0, interaction.shape[1]): 
           
            if interaction[i, j] == 1:                      #for associated
                label1.append(interaction[i,j])             #label1= labels for association(1)
                link_number = link_number + 1               #no. of associated samples  
                #link_position.append([i, j])     
                circRNA_fea_tmp = list(circRNA_fea[i]) 
                disease_fea_tmp = list(disease_fea[j])
                tmp_fea = (circRNA_fea_tmp,disease_fea_tmp)   #concatnated feature vector for an association
                train.append(tmp_fea)                       #train contains feature vectors of all associated samples
            elif interaction[i,j] == 0:                     #for no association
                label2.append(interaction[i,j])             #label2= labels for no association(0)
                #nonlink_number = nonlink_number + 1
                #nonLinksPosition.append([i, j])  
                circRNA_fea_tmp1 = list(circRNA_fea[i])		               
                disease_fea_tmp1 = list(disease_fea[j])
                test_fea= (circRNA_fea_tmp1,disease_fea_tmp1) #concatenated feature vector for not having association
                testfnl.append(test_fea)                    #testfnl contains feature vectors of all non associated samples
    
    
    print("link_number",link_number)
    
    
    #m = np.arange(len(label2))
    m = np.arange(445)
    np.random.shuffle(m)
    
    for x in m:
        ttfnl.append(testfnl[x])
        label22.append(label2[x])
             
    for x in range(0, 445):                         #for equalizing positive and negative samples
        tfnl= ttfnl[x]                                    
        lab= label22[x]                                      
        train.append(tfnl)                                  
        label1.append(lab)                                   
        
    
    return np.array(train), label1, np.array(testfnl)   

def calculate_performace(test_num, pred_y,  labels): #pred_y = proba, labels = real_labels
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    
    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score=0
        sensitivity =  float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
    else:
        precision = float(tp)/(tp+ fp)
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        f1_score= float(2*tp)/((2*tp)+fp+fn)

    return acc, precision, sensitivity, specificity, MCC,f1_score

def transfer_array_format(data):    #data=X  , X= all the miRNA features, disease features 
    formated_matrix1 = []
    formated_matrix2 = []
    #pdb.set_trace()
    #pdb.set_trace()
    for val in data:
        #formated_matrix1.append(np.array([val[0]]))
        formated_matrix1.append(val[0])   #contains miRNA features ?
        formated_matrix2.append(val[1])   #contains disease features ?
        #formated_matrix1[0] = np.array([val[0]])
        #formated_matrix2.append(np.array([val[1]]))
        #formated_matrix2[0] = val[1]      
    
    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def DNN():
    model = Sequential()
    model.add(Dense(input_dim=128, output_dim=500))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(input_dim=500, output_dim=500,init='glorot_normal'))   # Hidden layer1
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(input_dim=500, output_dim=300))#,init='glorot_normal'))  # Hidden layer 2
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(input_dim=300, output_dim=2))#,init='glorot_normal'))  #  Output layer
    model.add(Activation('sigmoid'))
    #sgd = SGD(l2=0.0,lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta) 
    return model



def DNN_auto(x_train):

    encoding_dim = 128
    input_img = Input(shape=(450,))


    encoded = Dense(350, activation='relu')(input_img)   #  (input layer)
    encoded = Dense(250, activation='relu')(encoded)     #  (hidden layer1)
    encoded = Dense(100, activation='relu')(encoded)     #  (hidden layer2)
    encoder_output = Dense(encoding_dim)(encoded)        # 128 - output (encoding layer)
    print()
    # decoder layers
    decoded = Dense(100, activation='relu')(encoder_output)
    decoded = Dense(250, activation='relu')(decoded)
    decoded = Dense(350, activation='relu')(decoded)
    decoded = Dense(450, activation='tanh')(decoded)

    autoencoder = Model(input=input_img, output=decoded)

    encoder = Model(input=input_img, output=encoder_output)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(x_train, x_train,epochs=20,batch_size=100,shuffle=True)  # second x_train is given instead of train labels in DNN, ie here, i/p=o/p


    encoded_imgs = encoder.predict(x_train)

    return encoder_output,encoded_imgs





def DeepCDA():
    
    X, labels,T = prepare_data(seperate = True)     #X= array of concatinated features,labels=corresponding labels
      
    X_data1, X_data2 = transfer_array_format(X) # X-data1 = circRNA features,  X_data2 = disease features 
    
    
    print (X_data1.shape,X_data2.shape)  
        
    
    X_data1= np.concatenate((X_data1, X_data2 ), axis = 1)
    
    
    print (X_data1.shape)  
       
           
    y, encoder = preprocess_labels(labels)
    num = np.arange(len(y))   #num is assigned an array like num = [0,1,2...len(y)]
    np.random.shuffle(num)
    X_data1 = X_data1[num]
    X_data2 = X_data2[num]
    y = y[num]
    
    t=0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    
    
    encoder,X_data1 = DNN_auto(X_data1) # X_data1 is assigned Auto encoded output
          
        
    num_cross_val = 5
    all_performance = []
    all_performance_rf = []
    all_performance_bef = []
    all_performance_DNN = []
    all_performance_SDADNN = []
    all_performance_blend = []
    all_labels = []
    all_prob = {}
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_averrage = []
    for fold in range(num_cross_val):
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        #train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        #test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        
          
        real_labels = []
        for val in test_label:
            if val[0] == 1:             
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        class_index = 0
        
            
                      

        class_index = class_index + 1
    
        prefilter_train = train1
        prefilter_test = test1
        
        #clf = svm.SVC(kernel='poly',degree=3,gamma=2,probability=True)
        #clf = RandomForestClassifier(n_estimators=100)
        
             
        model_DNN = DNN()
        
         #encoder,encoder_imgs=DNN_auto(prefilter_train)
        
        #print(encoder.shape)
        print(X_data1.shape)
        #print(X_data2.shape)
       
#        
        train_label_new_forDNN = np.array([[0,1] if i == 1 else [1,0] for i in train_label_new])
        print (train_label_new_forDNN[0])
       
        model_DNN.fit(prefilter_train,train_label_new_forDNN,batch_size=200,nb_epoch=20,shuffle=True) #training
        
               
        
        proba = model_DNN.predict_classes(prefilter_test,batch_size=200,verbose=True) #predicting classes
        #proba2 = model_DNN.predict(prefilter_test,batch_size=200,verbose=True) #predicting probabilities           
                            
                                                        
        ae_y_pred_prob = model_DNN.predict_proba(prefilter_test,batch_size=200,verbose=True)
               
        
        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(real_labels), proba,  real_labels)
      
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob[:,1])
        auc_score = auc(fpr, tpr)
       


        
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob[:,1])
        aupr_score = auc(recall, precision1)
        print ("AUTO-DNN:",acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score,f1_score)
        all_performance_DNN.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score,f1_score])
        t =t+1
        
        pyplot.plot(fpr,tpr,label= 'ROC fold %d (AUC = %0.4f)' % (t, auc_score))
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
                
        
            
        pyplot.xlabel('False positive rate, (1-Specificity)')
        pyplot.ylabel('True positive rate,(Sensitivity)')
        pyplot.title('Receiver Operating Characteristic curve: 5-Fold CV')
        pyplot.legend()
    
    mean_tpr /= num_cross_val
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    
    
       
    pyplot.plot(mean_fpr, mean_tpr,'--' ,linewidth=2.5,label='Mean ROC (AUC = %0.4f)' % mean_auc)
    pyplot.legend()

    
    pyplot.show()
    
    
    
    
    print('*******AUTO-RF*****')   
    print ('mean performance of AE-DNN ')
    print (np.mean(np.array(all_performance_DNN), axis=0))
    Mean_Result=[]
    Mean_Result= np.mean(np.array(all_performance_DNN), axis=0)
    print ('---' * 20)
    print('Mean-Accuracy=', Mean_Result[0],'\n Mean-precision=',Mean_Result[1])
    print('Mean-Sensitivity=', Mean_Result[2], '\n Mean-Specificity=',Mean_Result[3])
    print('Mean-MCC=', Mean_Result[4],'\n' 'Mean-auc_score=',Mean_Result[5])
    print('Mean-Aupr-score=', Mean_Result[6],'\n' 'Mean_F1=',Mean_Result[7])
    print ('---' * 20)
    
       
   

def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in proba]
    return label


if __name__=="__main__":
    DeepCDA()
