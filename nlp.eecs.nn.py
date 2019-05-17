#!/usr/bin/env python
# coding: utf-8

# In[2]:


##This file contains the neural network modelling of readmission after discharge. The first chunk is basic preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
df_admits = pd.read_csv('/Users/11kolop/MIMIC-III/ADMISSIONS.csv')
df_notes = pd.read_csv('/Users/11kolop/MIMIC-III/NOTEEVENTS.csv')
df_admits.ADMITTIME = pd.to_datetime(df_admits.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_admits.DISCHTIME = pd.to_datetime(df_admits.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_admits.DEATHTIME = pd.to_datetime(df_admits.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_admits = df_admits.sort_values(['SUBJECT_ID','ADMITTIME'])
df_admits = df_admits.reset_index(drop = True)
##Create admission time variable, to be used to create binary target variable later
df_admits['NEXT_ADMIT'] = df_admits.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
df_admits['NEXT_TYPE'] = df_admits.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)
#Do not use elective readmissions
rows = df_admits.NEXT_TYPE == 'ELECTIVE'
df_admits.loc[rows,'NEXT_ADMIT'] = pd.NaT
df_admits.loc[rows,'NEXT_TYPE'] = np.NaN
df_admits = df_admits.sort_values(['SUBJECT_ID','ADMITTIME'])
df_admits[['NEXT_ADMIT','NEXT_TYPE']] = df_admits.groupby(['SUBJECT_ID'])[['NEXT_ADMIT','NEXT_TYPE']].fillna(method = 'bfill')
df_admits['DAYS']=  (df_admits.NEXT_ADMIT - df_admits.DISCHTIME).dt.total_seconds()/(24*60*60)
#Only utilize discharge notes
df_notes_dis = df_notes.loc[df_notes.CATEGORY == 'Discharge summary']
df_notes_last = (df_notes_dis.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
#Merge admissions and notes charts
df_adnotes = pd.merge(df_admits[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS','NEXT_ADMIT','ADMISSION_TYPE','DEATHTIME']],
                        df_notes_last[['SUBJECT_ID','HADM_ID','TEXT']], 
                        on = ['SUBJECT_ID','HADM_ID'],
                        how = 'left')
df_adnotes.groupby('ADMISSION_TYPE').apply(lambda g: g.TEXT.isnull().sum())/df_adnotes.groupby('ADMISSION_TYPE').size()
#Create Target variable
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
import nltk
from nltk import word_tokenize
import string
#Remove stopwords, stem
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
sw = ['the','and','to','of','was','with','a','on','in','for','name',              
      'is','patient','s','he','at','as','or','one','she','his','her','am',                 
      'were','you','pt','pm','by','be','had','your','this','date',                
      'from','there','an','that','p','are','have','has','h','but','o',                
      'namepattern','which','every','also','should','if','it','been','who','during', 'x']
stemmer = SnowballStemmer("english")
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
#Vectorize and fit training and test text
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(tokenizer = tokenizer_better, stop_words =sw, min_df = 5, max_df =.9,)
vect.fit(df_train.TEXT.values.astype('U'))
dictionary = vect.vocabulary_.items() 
X_train_tf = vect.transform(df_train.TEXT.values.astype('U'))
X_test_tf = vect.transform(df_test.TEXT.values.astype('U'))
X_valid_tf = vect.transform(df_valid.TEXT.values.astype('U'))
y_train = df_train.OUTPUT_LABEL
y_test = df_test.OUTPUT_LABEL
y_valid = df_valid.OUTPUT_LABEL


# In[8]:


#view training and validation metrics
import matplotlib.pyplot as plt
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['auc']
    val_acc = history.history['val_auc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training auc')
    plt.plot(x, val_acc, 'r', label='Validation auc')
    plt.title('Training and validation AUC')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced'
                                               ,np.unique(y_train)
                                               ,y_train)


# In[9]:


#Trying a Simple neural network
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=len(vect.get_feature_names())))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
from keras.optimizers import Adam
Adam = Adam(lr=0.00001)
import tensorflow as tf
from keras import backend as K
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
model.compile(loss = "binary_crossentropy", optimizer = Adam ,metrics=[auc])
model.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=75)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history = model.fit(X_train_tf, y_train, 
          epochs=500, batch_size=64, verbose=1, 
          validation_data=(X_valid_tf, y_valid),callbacks=[es,mc])
saved_model = load_model('best_model.h5',custom_objects={ 'auc': auc })
scores = saved_model.evaluate(X_test_tf, y_test, verbose=1)
print("AUC:", scores[1]) 
plot_history(history)


# In[ ]:


#predictions, confusion matrix, and ROC curve
predictions_NN_prob = saved_model.predict(X_test_tf)
predictions_NN_prob = predictions_NN_prob[:,0]
predictions_NN_01 = np.where (predictions_NN_prob >.5,1,0)
from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion_matrix(y_test, predictions_NN_01)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions_NN_01, normalize=False) / float(y_test.size))
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()
cm = confusion_matrix(y_test, predictions_NN_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


# In[126]:


#Trying a Convolutional Neural Network with word sequences
word2idx = {word: idx for idx, word in enumerate(vect.get_feature_names())}
tokenize = vect.build_tokenizer()
preprocess = vect.build_preprocessor()
def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes
X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in df_train.TEXT]
MAX_SEQ_LENGTH = len(max(X_train_sequences, key=len))
print("MAX_SEQ_LENGTH=", MAX_SEQ_LENGTH)
from keras.preprocessing.sequence import pad_sequences
N_FEATURES = len(vect.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
print(X_train_sequences[0])
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding
model = Sequential()
model.add(Embedding(len(vect.get_feature_names()) + 1,
                    64,  
                    input_length=MAX_SEQ_LENGTH))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
from keras.optimizers import Adam
Adam = Adam(lr=0.00001)
import tensorflow as tf
from keras import backend as K
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
model.compile(loss='binary_crossentropy', optimizer= Adam , metrics=[auc])
print(model.summary())
X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in df_test.TEXT]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
X_valid_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in df_valid.TEXT]
X_valid_sequences = pad_sequences(X_valid_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
mc = ModelCheckpoint('best_model_one.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
hist = model.fit(X_train_sequences, y_train, 
          epochs=500, batch_size=64, verbose=1,
          validation_data=(X_valid_sequences, y_valid),class_weight = class_weight,callbacks=[es,mc])
saved_model_one = load_model('best_model_one.h5',custom_objects={ 'auc': auc })
scores = saved_model_one.evaluate(X_test_sequences, y_test, verbose=1)
print("AUC:", scores[1]) 
plot_history(hist)


# In[127]:


#predictions, confusion matrix, and ROC curve for the second model
predictions_NN_prob = saved_model_one.predict(X_test_sequences)
predictions_NN_prob = predictions_NN_prob[:,0]
predictions_NN_01 = np.where (predictions_NN_prob >.5,1,0)
from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion_matrix(y_test, predictions_NN_01)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions_NN_01, normalize=False) / float(y_test.size))
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()
cm = confusion_matrix(y_test, predictions_NN_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


# In[125]:


#Attempting an LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
 
model = Sequential()
model.add(Embedding(len(vect.get_feature_names()) + 1,
                    128, 
                    input_length=MAX_SEQ_LENGTH))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
from keras.optimizers import Adam
Adam = Adam(lr=0.00003)
import tensorflow as tf
from keras import backend as K
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
model.compile(loss='binary_crossentropy', optimizer= Adam, metrics=[auc])
print(model.summary())
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model_two.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
hi = model.fit(X_train_sequences, y_train, 
          epochs=50, batch_size=32, verbose=1, 
          validation_data=(X_valid_sequences, y_valid),class_weight = class_weight,callbacks=[es,mc])
saved_model_two = load_model('best_model_two.h5',custom_objects={ 'auc': auc })
scores = saved_model_two.evaluate(X_test_sequences, y_test, verbose=1)
print("AUC:", scores[1]) 

plot_history(hi)


# In[128]:


#predictions, confusion matrix, and ROC curve for the third model
predictions_NN_prob = saved_model_two.predict(X_test_sequences)
predictions_NN_prob = predictions_NN_prob[:,0]
predictions_NN_01 = np.where (predictions_NN_prob >.5,1,0)
from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion_matrix(y_test, predictions_NN_01)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions_NN_01, normalize=False) / float(y_test.size))
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()
cm = confusion_matrix(y_test, predictions_NN_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

