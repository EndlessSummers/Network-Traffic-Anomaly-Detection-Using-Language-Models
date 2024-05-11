# Network-Traffic-Anomaly-Detection-Using-Language-Models
## 📦 Overview
In this study, we utilized language models like CLSTM and Transformer to solve network traffic anomaly detection problem in cybersecurity. We employed techniques in both NLP and cybersecurity fields, such as Dyad-hour, sliding windows and tokenization to implement the system. Experiments are conducted on both supervised and unsupervised learning methods. We finally achieved an accuracy of 0.96 and F1 score of 0.89 on supervised method and a ROC score of 0.82 on unsupervised method, which is comparable to the performance of traditional detection systems, and overcame the limitations of a priori knowledge.

## ⚙️ Prerequisites

- Python 3.8.8
- torch 1.10.0+cu113
- torchvision 0.11.1+cu113
- pytorch_optimizer
- numpy
- pandas
- collections

## 🏁 Description of Files in the Repo

- supervised_CLSTM.ipynb : Experiment on supervised learning using CLSTM, including tests on different token choices: Protobyte and Service Port.
- supervised_Transformer.ipynb : Experiment on supervised learning using Transformer.
- unsupervised_CLSTM.ipynb : Experiment on unsupervised learning using CLSTM, including tests on different dataset: "Dirty", "Clean", "NoDDoS".


## 📊 Results
| Sr. No. | Model Name   | Accuracy | Precision       | Recall   | F1-Score | AUC-Score |                                                                                                                                                      |
|---------|--------------|-------------------------------------|-----------------|------|--------------|---------------|
| 1       | CLSTM (Protobyte)   | 0.95                           | 0.91   | 0.84  | 0.87         | --          | 
| 2       | CLSTM (Protobyte)   | 0.95                           | 0.91   | 0.84  | 0.87         | --          | 
| 1       | CLSTM (Protobyte)   | 0.95                           | 0.91   | 0.84  | 0.87         | --          | 
| 1       | CLSTM (Protobyte)   | 0.95                           | 0.91   | 0.84  | 0.87         | --          | 
