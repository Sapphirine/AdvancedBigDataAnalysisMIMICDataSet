#!/usr/bin/env python
# coding: utf-8

# In[1]:


##This file contains the basic exploration and modelling of readmission after discharge. The first chunk is basic preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from xgboost import XGBClassifier
nltk.download('punkt')
df_admits = pd.read_csv('/Users/11kolop/MIMIC-III/ADMISSIONS.csv')
df_notes = pd.read_csv('/Users/11kolop/MIMIC-III/NOTEEVENTS.csv')


# In[2]:


df_admits.ADMITTIME = pd.to_datetime(df_admits.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_admits.DISCHTIME = pd.to_datetime(df_admits.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_admits.DEATHTIME = pd.to_datetime(df_admits.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_admits = df_admits.sort_values(['SUBJECT_ID','ADMITTIME'])
df_admits = df_admits.reset_index(drop = True)
##Create admission time and type variables, to be used to create binary target variable later
df_admits['NEXT_ADMIT'] = df_admits.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
df_admits['NEXT_TYPE'] = df_admits.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)
#Do not use elective readmissions, instead shift to next unplanned readmission
rows = df_admits.NEXT_TYPE == 'ELECTIVE'
df_admits.loc[rows,'NEXT_ADMIT'] = pd.NaT
df_admits.loc[rows,'NEXT_TYPE'] = np.NaN
df_admits = df_admits.sort_values(['SUBJECT_ID','ADMITTIME'])
#Backfill all empty next admission entries with the following unplanned readmission
df_admits[['NEXT_ADMIT','NEXT_TYPE']] = df_admits.groupby(['SUBJECT_ID'])[['NEXT_ADMIT','NEXT_TYPE']].fillna(method = 'bfill')
df_admits['DAYS']=  (df_admits.NEXT_ADMIT - df_admits.DISCHTIME).dt.total_seconds()/(24*60*60)


# In[3]:


#Only utilize discharge notes
df_notes_dis = df_notes.loc[df_notes.CATEGORY == 'Discharge summary']
df_notes_last = (df_notes_dis.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
#Merge admissions and notes charts to link on subject and admission id
df_adnotes = pd.merge(df_admits[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS','NEXT_ADMIT','ADMISSION_TYPE','DEATHTIME']],
                        df_notes_last[['SUBJECT_ID','HADM_ID','TEXT']], 
                        on = ['SUBJECT_ID','HADM_ID'],
                        how = 'left')
df_adnotes.groupby('ADMISSION_TYPE').apply(lambda g: g.TEXT.isnull().sum())/df_adnotes.groupby('ADMISSION_TYPE').size()
#Create Target variable for admission less than 30 days after discharte
df_adnotes['OUTPUT_LABEL'] = (df_adnotes.DAYS < 30).astype('int')
df_adnotes = df_adnotes.sample(n = len(df_adnotes), random_state = 42)
df_adnotes = df_adnotes.reset_index(drop = True)
#Only analyze patients who did not die in hospital
no_death = df_adnotes.DEATHTIME.isnull()
df_not_death = df_adnotes.loc[no_death].copy()
df_not_death = df_not_death.sample(n = len(df_not_death), random_state = 42)
df_not_death = df_not_death.reset_index(drop = True)
#Training Validation Test Split
df_valid_test=df_not_death.sample(frac=0.20,random_state=42)
df_test = df_valid_test.sample(frac = 0.5, random_state = 42)
df_valid = df_valid_test.drop(df_test.index)
df_train_all=df_not_death.drop(df_valid_test.index)


# In[4]:


one_len = df_train_all[df_train_all['OUTPUT_LABEL'] == 1].shape[0]
zero_len = df_train_all[df_train_all['OUTPUT_LABEL'] == 0].shape[0]
plt.bar(10,one_len,3, label="ONE")
plt.bar(20,zero_len,3, label="ZERO")
plt.legend()
plt.ylabel('Number')
plt.title('Proportion')
plt.show()


# In[5]:


rows_pos = df_train_all.OUTPUT_LABEL == 1
df_train_pos = df_train_all.loc[rows_pos]
df_train_neg = df_train_all.loc[~rows_pos]
#Use undersampling of negative cases
df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 42)],axis = 0)
df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)
#Preprocess text
def preprocess_text(df):
    df.TEXT = df.TEXT.fillna(' ')
    df.TEXT = df.TEXT.str.replace('\n',' ')
    df.TEXT = df.TEXT.str.replace('\r',' ')
    df.TEXT = df.TEXT.str.replace('[^A-Za-z0-9(),!?@\'\`\"\_\n]', ' ')
    return df
df_train = preprocess_text(df_train)
df_test = preprocess_text(df_test)
df_valid = preprocess_text(df_valid)


# In[6]:


one_len = df_train[df_train['OUTPUT_LABEL'] == 1].shape[0]
zero_len = df_train[df_train['OUTPUT_LABEL'] == 0].shape[0]
plt.bar(10,one_len,3, label="ONE")
plt.bar(20,zero_len,3, label="ZERO")
plt.legend()
plt.ylabel('Number')
plt.title('Proportion')
plt.show()


# In[ ]:


import nltk
from nltk import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
sw = ['the','and','to','of','was','with','a','on','in','for','name',              
      'is','patient','s','he','at','as','or','one','she','his','her','am',                 
      'were','you','pt','pm','by','be','had','your','this','date',                
      'from','there','an','that','p','are','have','has','h','but','o',                
      'namepattern','which','every','also','should','if','it','been','who','during', 'x']
from nltk.stem.snowball import SnowballStemmer
sw = stopwords.words('english')
stemmer = SnowballStemmer("english")
#Remove stopwords, stem
def stemming(text):    
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 
df_train['TEXT'] = df_train['TEXT'].apply(stemming)
df_test['TEXT'] = df_test['TEXT'].apply(stemming)
df_valid['TEXT'] = df_valid['TEXT'].apply(stemming)
def tokenizer_better(text):  
    punc_list = string.punctuation+'0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens
from sklearn.feature_extraction.text import CountVectorizer
#Vectorize and fit training and test text
vect = CountVectorizer(lowercase=True, tokenizer = tokenizer_better,stop_words =sw)
vect.fit(df_train.TEXT.values.astype('U'))
dictionary = vect.vocabulary_.items() 
X_train_all_tf = vect.transform(df_train_all.TEXT.values.astype('U'))
X_train_tf = vect.transform(df_train.TEXT.values.astype('U'))
X_valid_tf = vect.transform(df_valid.TEXT.values.astype('U'))
y_train = df_train.OUTPUT_LABEL
y_train_all = df_train_all.OUTPUT_LABEL
y_valid = df_valid.OUTPUT_LABEL


# In[ ]:


#Most frequent words to occur
tokens = df_train.apply(lambda row: tokenizer_better(row['TEXT']), axis=1)
from nltk.probability import FreqDist
tokens =np.concatenate(tokens)
fdist = FreqDist(tokens)
lst = fdist.most_common(30)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count')


# In[134]:


#PCA approximation
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['red','blue','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            red_patch = mpatches.Patch(color='red', label='Negative')
            green_patch = mpatches.Patch(color='blue', label='Positive')
            plt.legend(handles=[red_patch, green_patch], prop={'size': 30})
fig = plt.figure(figsize=(16, 16))          
plot_LSA(X_train_tf, y_train)
plt.show()


# In[135]:


#Confusion Matrix function
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)  
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)  
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    return plt


# In[136]:


#Returning largest and smallest coefficient values from a logistic model
def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}   
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes


# In[137]:


#Plotting the words from the above function
def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]   
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]  
    fig = plt.figure(figsize=(10, 10))  
    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title('Negative', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Positive', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)    
    plt.subplots_adjust(wspace=0.8)
    plt.show()


# In[138]:


#Logistic Regression Model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(C = 0.0001, penalty = 'l2', random_state = 42)
clf.fit(X_train_tf, y_train)
model = clf
y_train_preds = model.predict_proba(X_train_tf)[:,1]
y_valid_preds = model.predict_proba(X_valid_tf)[:,1]
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
decisions = (y_valid_preds >= .5).astype(int)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, decisions)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Negative','Positive'], normalize=False, title='Confusion matrix')
plt.show()
print(cm)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_valid, decisions, normalize=False) / float(y_valid.size))
print(classification_report(y_valid, decisions))
fpr, tpr, _ = roc_curve(y_valid, decisions)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[139]:


#Important word coefficients from logistic model
importance = get_most_important_features(vect, clf, 25)
top_scores = [a[0] for a in list(importance.values())[0]['tops']]
top_words = [a[1] for a in list(importance.values())[0]['tops']]
bottom_scores = [a[0] for a in list(importance.values())[0]['bottom']]
bottom_words = [a[1] for a in list(importance.values())[0]['bottom']]
plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")


# In[121]:


#XGBoost Model Parameter Tuning

from sklearn.model_selection import GridSearchCV
model = XGBClassifier()
param_dist = {"max_depth": [10,30,50],
              "min_child_weight" : [1,3,6],
              "n_estimators": [200],
              "learning_rate": [0.05, 0.1,0.16],}
grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 
                                   verbose=10, n_jobs=-1)
grid_search.fit(X_train_tf, y_train)
grid_search.best_estimator_


# In[140]:


#XGBoost Model
model = XGBClassifier()
model.fit(X_train_tf, y_train)
y_pred = model.predict(X_valid_tf)
pres = [round(value) for value in y_pred]
cm1 = confusion_matrix(y_valid, pres)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm1, classes=['Negative','Positive'], normalize=False, title='Confusion matrix')
plt.show()
print(cm)
print(accuracy_score(y_valid, pres, normalize=False) / float(y_valid.size))
print(classification_report(y_valid, pres))
fpr, tpr, _ = roc_curve(y_valid, pres)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[141]:


#Random Forest Model 
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=1000)
clf.fit(X_train_tf,y_train)
y_pred=clf.predict(X_valid_tf)
cm2 = confusion_matrix(y_valid, y_pred)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm2, classes=['Negative','Positive'], normalize=False, title='Confusion matrix')
plt.show()
print(cm)
print(accuracy_score(y_valid, y_pred, normalize=False) / float(y_valid.size))
print(classification_report(y_valid, y_pred))
fpr, tpr, _ = roc_curve(y_valid, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[88]:


#SVM Model Parameter Tuning
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
def svc_param_selection(X, y, nfolds):
    Cs = [.001,.01,0.1, 1, 10, 100, 1000]
    gammas = [.001,.01,0.1, 1, 10, 100]
    param_grid = {'C': Cs, 'gamma' : gammas}
    svc=SVC(kernel='rbf')
    grid_search = GridSearchCV(svc, param_grid, cv=nfolds, scoring='roc_auc')
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_
svc_param_selection(X_train_tf,y_train,nfolds=5)


# In[142]:


#SVM Model 
from sklearn.svm import SVC 
svm = SVC(kernel='rbf') 
svm.fit(X_train_tf, y_train) 
y_pr=svm.predict(X_valid_tf)
cm3 = confusion_matrix(y_valid, y_pr)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm3, classes=['Negative','Positive'], normalize=False, title='Confusion matrix')
plt.show()
print(cm)
print(accuracy_score(y_valid, y_pr, normalize=False) / float(y_valid.size))
print(classification_report(y_valid, y_pr))
fpr, tpr, _ = roc_curve(y_valid, y_pr)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[143]:


#AdaBoost Model
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1,)
model = abc.fit(X_train_tf, y_train)
y_p = model.predict(X_valid_tf)
cm4 = confusion_matrix(y_valid, y_p)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm4, classes=['Negative','Positive'], normalize=False, title='Confusion matrix')
plt.show()
print(cm)
print(accuracy_score(y_valid, y_p, normalize=False) / float(y_valid.size))
print(classification_report(y_valid, y_p))
fpr, tpr, _ = roc_curve(y_valid, y_p)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

