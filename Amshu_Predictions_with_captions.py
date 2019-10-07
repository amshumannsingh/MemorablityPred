#!/usr/bin/env python
# coding: utf-8

# # Memorability Prediction

# In[ ]:


#Spearman coefficient function
def Get_score(Y_pred,Y_true):
    '''Calculate the Spearmann"s correlation coefficient'''
    Y_pred = np.squeeze(Y_pred)
    Y_true = np.squeeze(Y_true)
    if Y_pred.shape != Y_true.shape:
        print('Input shapes don\'t match!')
    else:
        if len(Y_pred.shape) == 1:
            Res = pd.DataFrame({'Y_true':Y_true,'Y_pred':Y_pred})
            score_mat = Res[['Y_true','Y_pred']].corr(method='spearman',min_periods=1)
            print('The Spearman\'s correlation coefficient is: %.3f' % score_mat.iloc[1][0])
        else:
            for ii in range(Y_pred.shape[1]):
                Get_score(Y_pred[:,ii],Y_true[:,ii])


# In[111]:


#Mapping notebook to drive
from google.colab import drive
import os
drive.mount('/content/drive/')
os.chdir('/content/drive/My Drive/ML Assignment/dev-set/')
get_ipython().system('pwd')


# In[112]:


get_ipython().system('pip install pyprind')


# In[260]:


import pandas as pd
from keras import Sequential
from keras import layers
from keras import regularizers
from keras import optimizers
import numpy as np
from string import punctuation
import pyprind
from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import nltk #for Stopwords
from nltk.corpus import stopwords
from natsort import natsorted,ns #for sorting files in C3D and HMP
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))


# In[ ]:



# for reproducability
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)


# 
# 
# 
# ## Defining Functions for reading **data** here 

# In[ ]:


def read_HMP(fname):
    """Scan HMP(Histogram of Motion Patterns) features from file"""
    with open(fname) as f:
        for line in f:
            pairs=line.split()
            HMP_temp = { int(p.split(':')[0]) : float(p.split(':')[1]) for p in pairs}
    # there are 6075 bins, fill zeros
    HMP = np.zeros(6075)
    for idx in HMP_temp.keys():
        HMP[idx-1] = HMP_temp[idx]            
    return HMP


# In[ ]:


#read C3D
def read_C3D(fname):
    with open(fname) as f:
        for line in f:
            C3D =[float(item) for item in line.split()] 
    return C3D


# In[ ]:


#defining column for video ID
def vname2ID(vnames):
    vid = [ os.path.splitext(vn)[0][5:] for vn in vnames]
    return vid


# In[ ]:


#Creating vid column
Feat_path = '/content/drive/My Drive/ML Assignment/dev-set/'
vnames = os.listdir(Feat_path+'HMP')
vid = vname2ID(vnames)


# In[120]:


print(vid)


# In[121]:


#video ID values for training set
vid1 = map(int, vid)
vid1 = list(map(int, vid))
vid1.sort()
print(vid1)


# In[ ]:


Feat_path1 = '/content/drive/My Drive/ML Assignment/test-set/'
#Creating vid column
vnames = os.listdir(Feat_path1+'HMP_test')
vid_test = vname2ID(vnames)


# In[123]:


#video ID values for test set
vid_test1 = map(int, vid_test)
vid_test1 = list(map(int, vid_test))
vid_test1.sort()
print(vid_test1)


# ## Loading the captions and the memorability scores

# In[ ]:


# load labels and captions
def read_caps(fname):
    """Load the captions into a dataframe"""
    vn = []
    cap = []
    df = pd.DataFrame();
    with open(fname) as f:
        for line in f:
            pairs = line.split()
            vn.append(pairs[0])
            cap.append(pairs[1])
        df['video']=vn
        df['caption']=cap
    return df


# load the captions
cap_path = './dev-set_video-captions.txt'
df_cap=read_caps(cap_path)

# load the ground truth values
label_path = './'
labels=pd.read_csv(label_path+'dev-set_ground-truth.csv')


# In[ ]:


#Loading for test set
# load the captions
cap_path1 = '/content/drive/My Drive/ML Assignment/test-set/test-set-1_video-captions.txt'
df_cap_test=read_caps(cap_path1)

# load the ground truth values
label_path = '/content/drive/My Drive/ML Assignment/test-set/'
labels_test=pd.read_csv(label_path+'ground_truth_template.csv')


# In[126]:


print(df_cap)


# In[127]:


print(labels)


# In[128]:


print(labels_test)


# In[129]:


#Combing traing and test dataframes for caption
df_cap_comb = df_cap.append(df_cap_test, ignore_index=True)
print(df_cap_comb)
print(df_cap_comb.shape)


# In[148]:


#Removing stop words from the dataset
df_cap_comb['caption'] = df_cap_comb['caption'].apply(lambda x: '-'.join([word for word in x.split() if word not in (stop_words)]))
print(df_cap_comb)


# In[149]:


#Finding count for each word in the captions column 
counts = Counter()
# setup prograss tracker
pbar = pyprind.ProgBar(len(df_cap_comb['caption']), title='Counting word occurrences')
for i, cap in enumerate(df_cap_comb['caption']):
    # replace punctuations with space
    # convert words to lower case 
    text = ''.join([c if c not in punctuation else ' ' for c in cap]).lower()
    df_cap_comb.loc[i,'caption'] = text
    pbar.update()
    counts.update(text.split())


# In[150]:


print(counts)
print(len(counts))


# In[152]:


df_cap_comb.head()


# ## Calling functions to obtain C3D and HMP Features in Dataframe

# In[ ]:


#getting video names using features' name
def getnames(featurename):
  vnames = os.listdir(Feat_path+featurename)
  return vnames

#get full path of the features using features' name 
def getpaths(featurename):
  fpath = [Feat_path+featurename+'/' + x for x in  getnames(featurename)]
  sorted = natsorted(fpath,alg=ns.IGNORECASE)
  print('Feature Path:')
  print(sorted[:5])
  return sorted


# In[ ]:


#Loading for test set
#getting video names using features' name
def getnames1(featurename):
  vnames = os.listdir(Feat_path1+featurename)
  return vnames

#get full path of the features using features' name 
def getpaths1(featurename):
  fpath = [Feat_path1+featurename+'/' + x for x in  getnames1(featurename)]
  sorted = natsorted(fpath,alg=ns.IGNORECASE)
  print('Feature Path:')
  print(sorted[:5])
  return sorted


# In[ ]:


#load C3D and return as an array of list
def C3D():
  path = getpaths('C3D')
  c3d = []
  print('loading C3D')
  for item in path:
    c3d.append(read_C3D(item))
  print('done')
  return np.asarray(c3d)

#load HMP and return as an array of list
def HMP():
  path = getpaths('HMP')
  hmp = []
  print('loading HMP')
  for item in path:
    hmp.append(read_HMP(item))
  print('done')
  return np.asarray(hmp)


# In[ ]:


#test set extraction
#load C3D and return as an array of list
def C3D1():
  path = getpaths1('C3D_test')
  c3d = []
  print('loading C3D')
  for item in path:
    c3d.append(read_C3D(item))
  print('done')
  return np.asarray(c3d)

#load HMP and return as an array of list
def HMP1():
  path = getpaths1('HMP_test')
  hmp = []
  print('loading HMP')
  for item in path:
    hmp.append(read_HMP(item))
  print('done')
  return np.asarray(hmp)


# In[138]:


#Reading C3D into Dataframe
df_c3d=C3D()
print(df_c3d)
df_c3d.shape


# In[139]:


#Reading HMP into Dataframe
df_hmp=HMP()
print(df_hmp)
df_hmp.shape


# In[143]:


#test set C3D
df_c3d_test=C3D1()
print(df_c3d_test)
df_c3d_test.shape


# In[144]:


#test set HMP
df_hmp_test=HMP1()
print(df_hmp_test)
df_hmp_test.shape


# ##  Preprocessing the captions for model inputs
# ### Separate words and count each word's occurrence

# ### Maping each unique word to an integer (one-hot encoding)

# In[153]:


# build the word index
len_token = len(counts)
tokenizer = Tokenizer(num_words=len_token)
print(len_token)


# In[ ]:


tokenizer.fit_on_texts(list(df_cap_comb.caption.values)) #fit a list of captions to the tokenizer
#the tokenizer vectorizes a text corpus, by turning each text into either a sequence of integers 


# In[155]:


print(len(tokenizer.word_index))


# In[ ]:


one_hot_res = tokenizer.texts_to_matrix(list(df_cap_comb.caption.values),mode='binary')
sequences = tokenizer.texts_to_sequences(list(df_cap_comb.caption.values))


# In[157]:


#Just to visualise some stuff in sequences and counts
print(sequences[0]) # prints location of words from caption 0 'blonde woman is massaged tilt down'
print(counts['blonde']) # no. of occurences of 'blonde'
n=3
print('Least Common: ', counts.most_common()[:-n-1:-1])       # n least common elements
print('Most Common: ',counts.most_common(n))                     # n most common elements


# ### Making all the sequences same length by padding zeros from 1 to (N - len(seq))

# In[ ]:


# calculating max length
max_len = 50


# ### Making sequences index same length

# In[159]:


print(sequences[0]) # length of 1st sequence


# In[160]:


#Function to convert all caption tokens to length 50
X_seq = np.zeros((len(sequences),max_len))
for i in range(len(sequences)):
    n = len(sequences[i])
    if n==0:
        print(i)
    else:
        X_seq[i,-n:] = sequences[i]
X_seq.shape


# In[161]:


print(X_seq[5999,:])


# In[162]:


print(X_seq[0,:]) # length of 1st sequence after padding the caption with zeros.


# In[163]:


print(X_seq[6000,:])


# In[164]:


#splitting training and test dataset
X_seq1=X_seq[0:6000]
X_seq_test=X_seq[6000:8000]
print(X_seq1.shape)
print(X_seq_test.shape)
print(X_seq_test[0,:])


# Merging C3D, HMP and captions into one dataframe

# In[264]:


df_feat=np.concatenate((df_c3d,df_hmp),axis=1)
#df_feat=np.concatenate((df_feat,X_seq1),axis=1)
print(df_feat)
df_feat.shape


# In[263]:


#Merging for test Dataset
df_feat_test=np.concatenate((df_c3d_test,df_hmp_test),axis=1)
#df_feat_test=np.concatenate((df_feat_test,X_seq_test),axis=1)

print(df_feat_test[0])
df_feat_test.shape


# In[292]:


len_feat=len(df_feat[0])
print(len_feat)


# # Predicting video memorability using Combined Feature set 

# Throwing C3D and HMP features together into a Neural Network

# In[305]:


Y_train = labels[['short-term_memorability','long-term_memorability']].values
Y_test = labels_test[['short-term_memorability','long-term_memorability']].values
X_train = df_feat;
X_test = df_feat_test

#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# add dropout
# add regularizers
model = Sequential()
model.add(layers.Dense(2500,activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(6176,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3000,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3000,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(2500,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(2,activation='sigmoid'))

# compile the model 
model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
optimizers.RMSprop(lr=0.0001, rho=0.95, epsilon=None, decay=0.0)

# training the model 
history = model.fit(X_train,Y_train,epochs=30,validation_data=(X_test,Y_test))

# visualizing the model
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()


# In[306]:


#Prediction dataframe
predictions = model.predict(X_test)
print(predictions.shape)


# In[308]:


Get_score(predictions, Y_test)
print(predictions)


# Combining Predictions dataframe with video labels and writng to CSV

# In[334]:


y_pred=labels_test['video']
print(y_pred)


# In[357]:


pred = pd.DataFrame(({'Short-term memorability':predictions[:,0],'Long-term memorability':predictions[:,1]})) 
pred2= pd.concat((y_pred,pred),axis=1)
print(pred2)


# In[358]:


os.chdir('/content/drive/My Drive/ML Assignment/')
numpy.savetxt("predictions1.csv", pred2, delimiter=",")


# In[ ]:


model.save('/content/drive/My Drive/ML Assignment/model.h5')  # creates a HDF5 file 'my_model.h5'

