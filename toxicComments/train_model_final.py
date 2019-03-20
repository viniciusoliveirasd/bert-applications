#!/usr/bin/env python
# coding: utf-8

# # Improved LSTM baseline - Using BERT last layer word-embeddings
# 
# This file is a somewhat improved version of [Keras - Bidirectional LSTM baseline](https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-051) along with some additional documentation of the steps.

# In[1]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[2]:


path = 'input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
TRAIN_DATA_FILE=f'{path}{comp}train.csv'
TEST_DATA_FILE=f'{path}{comp}test.csv'


# Set some basic config parameters:

# In[3]:


embed_size = 768 # how big is each word vector
max_features = 30522 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 128 # max number of words in a comment to use


# Read in our data and replace missing values:

# In[4]:


train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)


print('Loading train and test set')
list_sentences_train = train_df["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train_df[list_classes].values
list_sentences_test = test_df["comment_text"].fillna("_na_").values
print('Datasets succ. loaded')

train_lines = list_sentences_train
test_lines = list_sentences_test
train_labels = y_train



# In[13]:


# In[6]:


BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = 'models/uncased_L-12_H-768_A-12'
print('BERT_MODEL dir',BERT_MODEL)
print(BERT_PRETRAINED_DIR)

# In[7]:

#Loading aux docs 
import modeling
import optimization
import run_classifier
import tokenization
import tensorflow as tf

print('Loaded auxiliary scripts')

def create_examples(lines, set_type, labels=None):
#Generate data for the BERT model
    guid = f'{set_type}'
    examples = []
    if guid == 'train':
        for line, label in zip(lines, labels):
            text_a = line
            label = str(label)
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    else:
        for line in lines:
            text_a = line
            label = '[0 0 0 0 0 0]'
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

# Model Hyper Parameters
tf.random.set_random_seed(49)

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 128
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000 #if you wish to finetune a model on a larger dataset, use larger interval
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
train_examples = create_examples(train_lines, 'train', labels=train_labels)
test_examples = create_examples(test_lines, 'test')

print('train_examples object loaded')


# In[9]:


num_train_steps = int(
    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

print('Steps calculated')

# In[10]:

#Auxiliary Functions to provide label_list for our problem
l = [] 
def return_l(n, l_r):
    if n == 1:
        l = l_r
        return l_r
    new_l = []
    for t in l_r:
        t.append(0)
        new_l.append(t[:])
        t.pop()
        t.append(1)
        new_l.append(t[:])
        t.pop()
    return return_l(n-1, new_l)

def return_str(n, l_r):
    if n == 1:
        new_l = []
        for t in l_r:
            new_l.append(t +']')
        l = new_l
        return new_l
    new_l = []
    for t in l_r:
        
        new_l.append(t +' 0')
        new_l.append(t +' 1')
    return return_str(n-1, new_l)


label_list_mult = return_str(6, ['[1','[0'])


# In[12]:


# Train the model.
print('Please wait..., loading train words in model')
train_features = run_classifier.convert_examples_to_features(train_examples, label_list_mult,  MAX_SEQ_LENGTH, tokenizer)
print('train_features loaded')
print('Please wait..., loading test words in model')
test_features = run_classifier.convert_examples_to_features(test_examples, label_list_mult,  MAX_SEQ_LENGTH, tokenizer)
print('test_features loaded')

# In[13]:


formOfList_label_list_mult = return_l(6, [[1], [0]])


# In[16]:

print('loading X_t and y_train')
X_t = []
y_train = []
for i, token in enumerate(train_features): 
    X_t.append(token.input_ids)
    #print(token.label_id)
    y_train.append(formOfList_label_list_mult[token.label_id])
X_t = np.asarray(X_t)
y_train = np.asarray(y_train)
print('loaded')

print('loading X_te')
X_te = []
for i, token in enumerate(test_features): 
    X_te.append(token.input_ids)
    #print(token.label_id)
X_te = np.asarray(X_te)
print('X_te loaded')


# In[19]:

print('importing torch bert  model...')
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForSequenceClassification
bert_model = BertModel.from_pretrained("bert-base-uncased",cache_dir="model")
embedding_matrix = []
for token in bert_model.embeddings.word_embeddings.parameters():
    embedding_matrix.append(token)
emb_ma = embedding_matrix[0].tolist()
array_emb_ma = np.asarray(emb_ma)

print('embedding matrix defined')
embedding_matrix = array_emb_ma


# Simple bidirectional LSTM with two fully connected layers. We add some dropout to the LSTM since even 2 epochs is enough to overfit.

# In[21]


# In[22]:


inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('model defined and compiled ')
# Now we're ready to fit out model! Use `validation_split` when not submitting.


model.fit(X_t, y_train, batch_size=32, epochs=2, validation_split=0.1);


# And finally, get predictions for the test set and prepare a submission CSV:

# In[ ]:





y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


