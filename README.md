# bert-benchmarks
Google's Bert algorithm study and real-life applications for Deep Learning project


In this project we will be studying, analyzing and exploring the recently launched Bidirectional Encoder Representations from Transformers (BERT). It was launched by Google AI in Oct 2018 in the following paper https://arxiv.org/abs/1810.04805. 

The original repository of Google Research can be found at : https://github.com/google-research/bert

We will also inspire our work and research in the pytorch implementation of BERT made by huggingface : https://github.com/huggingface/pytorch-pretrained-BERT/

In a first moment we have devoted all our efforts to better understand how it works and why it is considered a breakthrough in the NLP reasearch field. Some remarkable material can be found in the internet including the following ones :

* Bert paper: https://arxiv.org/pdf/1810.04805.pdf

* https://jalammar.github.io/illustrated-bert/

* https://jalammar.github.io/illustrated-transformer/


In a second step we will be fine-tuning, adapting and applying existing models to real-life applications as Kaggles competitions and the SQuAD Dataset. Some of the competitions we have applied and/or are intending to apply are: 


| Use Case  | Link | Our results |
|:---------:|:----:|:-----------:|
| Kaggle: Quora Insincere Questions Classification  | https://www.kaggle.com/c/quora-insincere-questions-classification | F-Score 0.70240 / Acc: 0.96 |
| Kaggle: WSDM Fake News detection|  https://www.kaggle.com/c/fake-news-pair-classification-challenge/ | 0.86535 |
| Kaggle: Toxic Comment Classification Challenge - Glove Comparation |  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/ |Mean column-wise Area under ROC-Curve // Glove: 0.97718 Vs Bert: 0.97922 |
| French transformation of word vectors |  https://www.kaggle.com/c/detecting-insults-in-social-commentary | Extracted Word vectors but did not implement a specific task |

Instructions to run the code:

## Quora Insincere Questions Classification:

To finetune BERT for this competition please <a href="https://www.kaggle.com/c/quora-insincere-questions-classification/data
" target="_blank">download the dataset from the Kaggle competition</a> and put it in a folder called `input` inside the directory containing the `bert-classification.ipynb` script. 

You should have installed pandas, numpy, sklearn, tensorflow, zipfile, matplotlib and tqdm.

It took 48 hours to train 3 epochs in an Azure VM: Standard NC24 (24 vcpus, 224 GB memory)


## Fake News Detection:

To finetune BERT for this competition please <a href="https://www.kaggle.com/c/fake-news-pair-classification-challenge/data
" target="_blank">download the dataset from the Kaggle competition</a> and put it in a folder called `input` inside the directory containing the `train.py` script. 

You should have installed pandas, numpy, sklearn, pytorch and pytorch-pretrained-bert.

`train.py` script based on the Kernel: https://www.kaggle.com/bbqlp33/bert-huggingface-pytorch-pretrained-bert

It took 7 hours to train 3 epochs in an Azure VM: Standard NC24 (24 vcpus, 224 GB memory)



Some tests were run in the GPU environment with TPU but results are not yet ready. 
