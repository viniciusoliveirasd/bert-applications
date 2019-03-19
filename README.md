# bert-benchmarks
Google's Bert algorithm study and real-life applications for Deep Learning project


In this project we will be studying, analyzing and exploring the recently launched Bidirectional Encoder Representations from Transformers (BERT). It was launched by Google in Oct 2018 in the following paper https://arxiv.org/abs/1810.04805. 

The original repository of Google Research can be found at : https://github.com/google-research/bert

We will also inspire our work and research in the pytorch implementation of BERT made by huggingface : https://github.com/huggingface/pytorch-pretrained-BERT/

In a first moment we have devoted all our efforts to better understand how it works and why it is considered a breakthrough in the NLP reasearch field. Some remarkable material can be found in the internet including the following ones :

* The Bert paper itself 

* https://jalammar.github.io/illustrated-bert/

* https://jalammar.github.io/illustrated-transformer/


In a second step we will be fine-tuning, adapting and applying existing models to real-life applications as Kaggles competitions and the SQuAD Dataset. Some of the competitions we have applied and/or are intending to apply are: 


| Use Case  | Link | Our results |
|:---------:|:----:|:-----------:|
| Kaggle: Quora Insincere Questions Classification  | https://www.kaggle.com/c/quora-insincere-questions-classification | F-Score 0.70240 / Acc: 0.96 |
|Fake News detection|  https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/71421 | -|
| Toxic Comment Classification Challenge - Glove Comparation |  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/ | -|
| French transformation of word vectors |  https://www.kaggle.com/c/detecting-insults-in-social-commentary | -|



For the moment we are using Azure VM instances with a Tesla K80 GPU. 
Some tests were run in the GPU environment with TPU but results are not yet ready. 
