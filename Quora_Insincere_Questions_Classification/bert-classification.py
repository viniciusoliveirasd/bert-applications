#!/usr/bin/env python
# coding: utf-8

# #                                                                                 BERT
# 
# BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.
# 
# Academic paper which describes BERT in detail and provides full results on a number of tasks can be found here: https://arxiv.org/abs/1810.04805.
# 
# Github account for the paper can be found here: https://github.com/google-research/bert
# 
# BERT is a method of pre-training language representations, meaning training of a general-purpose "language understanding" model on a large text corpus (like Wikipedia), and then using that model for downstream NLP tasks (like question answering). BERT outperforms previous methods because it is the first *unsupervised, deeply bidirectional *system for pre-training NLP.

# ![](https://www.lyrn.ai/wp-content/uploads/2018/11/transformer.png)

# 
# # Downloading all necessary dependencies
# You will have to turn on internet for that.
# 
# This code is slightly modefied version of this colab notebook https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb

# In[1]:


import pandas as pd
import os
import numpy as np
import zipfile
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import datetime


# In[2]:


#downloading weights and cofiguration file for the model
get_ipython().system('wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')


# In[2]:


repo = 'model_repo'
with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:
    zip_ref.extractall(repo)


# In[2]:


get_ipython().system("ls 'model_repo/uncased_L-12_H-768_A-12'")


# In[5]:


get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py ')
get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/optimization.py ')
get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/run_classifier.py ')
get_ipython().system('wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py ')


# Example below is done on preprocessing code, similar to **CoLa**:
# 
# The Corpus of Linguistic Acceptability is
# a binary single-sentence classification task, where 
# the goal is to predict whether an English sentence
# is linguistically “acceptable” or not
# 
# You can use pretrained BERT model for wide variety of tasks, including classification.
# The task of CoLa is close to the task of Quora competition, so I thought it woud be interesting to use that example.
# Obviously, outside sources aren't allowed in Quora competition, so you won't be able to use BERT to submit a prediction.
# 
# 
# 
# 

# In[3]:


# Available pretrained model checkpoints:
#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model
#We will use the most basic of all of them
BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = f'{repo}/uncased_L-12_H-768_A-12'
OUTPUT_DIR = f'{repo}/outputs'
print(f'***** Model output directory: {OUTPUT_DIR} *****')
print(f'***** BERT pretrained directory: {BERT_PRETRAINED_DIR} *****')


# In[4]:


from sklearn.model_selection import train_test_split

train_df =  pd.read_csv('input/train.csv')

train, test = train_test_split(train_df, test_size = 0.1, random_state=42)

train_lines, train_labels = train.question_text.values, train.target.values
test_lines, test_labels = test.question_text.values, test.target.values


# In[5]:


get_ipython().system(' ls model_repo/outputs/')
#! rm -r model_repo/outputs


# In[6]:


import modeling
import optimization
import run_classifier
import tokenization
import tensorflow as tf


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
            label = '0'
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 128
# Model configs
SAVE_CHECKPOINTS_STEPS = 10000 #if you wish to finetune a model on a larger dataset, use larger interval
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(OUTPUT_DIR, 'model.ckpt-26000')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

label_list = ['0', '1']
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
train_examples = create_examples(train_lines, 'train', labels=train_labels)

tpu_cluster_resolver = None #Since training will happen on GPU, we won't need a cluster resolver
#TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

num_train_steps = int(
    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available  
    use_one_hot_embeddings=True)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)


# In[ ]:


"""
Note: You might see a message 'Running train on CPU'. 
This really just means that it's running on something other than a Cloud TPU, which includes a GPU.
"""

# Train the model.
print('Please wait...')
train_features = run_classifier.convert_examples_to_features(
    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
print('***** Started training at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(len(train_examples)))
print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
tf.logging.info("  Num steps = %d", num_train_steps)
train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('***** Finished training at {} *****'.format(datetime.datetime.now()))


# In[7]:


"""
There is a weird bug in original code.
When predicting, estimator returns an empty dict {}, without batch_size.
I redefine input_fn_builder and hardcode batch_size, irnoring 'params' for now.
"""

def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        print(params)
        batch_size = 32

        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# In[8]:


predict_examples = create_examples(test_lines, 'test')

predict_features = run_classifier.convert_examples_to_features(
    predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

predict_input_fn = input_fn_builder(
    features=predict_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

result = estimator.predict(input_fn=predict_input_fn)


# In[9]:


from tqdm import tqdm
preds = []
for prediction in tqdm(result):
    for class_probability in prediction['probabilities']:
        preds.append(float(class_probability))

results = []
for i in tqdm(range(0,len(preds),2)):
    if preds[i] < 0.9:
        results.append(1)
    else:
        results.append(0)


# In[10]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print('%.2f' % accuracy_score(np.array(results), test_labels))
print('%.5f' % f1_score(np.array(results), test_labels))


# There are several downsides for BERT at this moment:
# 
# - Training is expensive. All results on the paper were fine-tuned on a single Cloud TPU, which has 64GB of RAM. It is currently not possible to re-produce most of the BERT-Large results on the paper using a GPU with 12GB - 16GB of RAM, because the maximum batch size that can fit in memory is too small. 
# 
# - At the moment BERT supports only English, though addition of other languages is expected.
# 
# 

# # Competition test
# 

# We've run a test with 30% of Quora data on Standard NC6 (6 vcpus, 56 GB memory) and achieved f1 score of 0.684.(11th place at the moment)
# 
# **You can't use BERT in the competition, the notebook will fail when it comes to real testing.**
# 
# Training took about 12 hours.
# Results are really amazing, espetially because it's a raw model on random 30% sample with no optimization or ensamble, using the simlest of 3 released models.
# 
# We didn't even have to preprocess anything, model does it for you.
# 
