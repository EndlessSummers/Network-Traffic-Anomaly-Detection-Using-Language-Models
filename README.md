# Network-Traffic-Anomaly-Detection-Using-Representation-Learning-and-Language-Models
## 📦 Overview
In this study, we utilized language models like CLSTM, Transformer and GPT-2 to solve network traffic anomaly detection problem in cybersecurity. We employed techniques in both NLP and cybersecurity fields, such as Dyad-hour, sliding windows and tokenization to implement the system. Experiments are conducted on both supervised and unsupervised learning methods. We finally achieved an accuracy of 0.96 and F1 score of 0.89 on supervised method and an accuracy of 0.95 and a F1-score of 0.95 on unsupervised method, which is comparable to the performance of traditional detection systems, and overcame the limitations of a priori knowledge.

## ⚙️ Prerequisites

- Python 3.8.8
- torch 1.10.0+cu113
- torchvision 0.11.1+cu113
- pytorch_optimizer
- numpy
- pandas

## 🏁 Description of Files in the Repo

- supervised_CLSTM.ipynb : Experiment on supervised learning using CLSTM, including tests on different token choices: Protobyte and Service Port.
- supervised_Transformer.ipynb : Experiment on supervised learning using Transformer.
- unsupervised_CLSTM.ipynb : Experiment on unsupervised learning using CLSTM, including tests on different dataset: "Dirty", "Clean", "NoDDoS".


## 📊 Results
| Sr. No. | Model Name                        | Accuracy | Precision       | Recall   | F1-Score | AUC-Score |
|---------|-----------------------------------|----------|-----------------|----------|----------|-----------|
| 1       | Supervised CLSTM (Protobyte)      | 0.95     | 0.91            | 0.84     | 0.87     | ---       | 
| 2       | Supervised CLSTM (Service Port)   | 0.96     | 0.93            | 0.14     | 0.24     | ---       | 
| 3       | Transformer                       | 0.96     | 0.93            | 0.86     | 0.89     | ---       | 
| 4       | Unsupervised CLSTM (Dirty)        | 0.82     | 0.76            | 0.71     | 0.73     | 0.75      | 
| 5       | Unsupervised CLSTM (Clean))       | 0.89     | 0.80            | 0.75     | 0.77     | 0.82      | 
| 6       | Unsupervised CLSTM (NoDDoS)       | 0.87     | 0.79            | 0.75     | 0.77     | 0.80      | 
