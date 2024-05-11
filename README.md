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

- Model_Weights_and_Eval_Metrics/ : Model weights trained with different hyperparameters and loss&acc for each epoch of training and testing.
- plots/ : Visualize acc and loss of models trained and tested using different hyperparameters in each epoch
- SE_ResNet_55.py : SEResNet model with 55 layers.
- SE_ResNet_68.py : SEResNet model with 68 layers.
- cifar_test_nolabels.pkl : A custom test dataset.
- main.ipynb : Train and test.


## 📊 Results
| Sr. No. | Model Name   | # Residual Blocks in Residual Layer | Optimizer       | lr   | Augmentation | Gradient Clip | Batch Size | Params | Test Acc | File Link                                                                                                                                                         |
|---------|--------------|-------------------------------------|-----------------|------|--------------|---------------|------------|--------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1       | SEResnet55   | [2,2,2,2]                           | Lookahead+SGD   | 0.1  | True         | True          | 32         | 4.99M  | 95.81%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/4residual_layers_model)                                       |
| 2       | SEResnet68   | [4,4,3]                             | Lookahead+SGD          | 0.1  | True         | True          | 32         | 4.70M  | 96.28%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/batch_size32_model)                                           |
| **3**   | **SEResnet68** | **[4,4,3]**                       | **Lookahead+SGD** | **0.1** | **True**     | **True**      | **128**      | **4.70M** | **96.48%** | [**LINK**](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/best_acc_model)                                          |
| 4       | SEResnet68   | [4,4,3]                             | Lookahead+SGD   | 0.01 | True         | True          | 32         | 4.70M  | 96.23%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/lr0.01_model)                                                 |
| 5       | SEResnet68   | [4,4,3]                             | Ranger   | 0.1  | True         | True          | 32         | 4.70M  | 95.67%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/sgd_model)                                                    |
| 6       | SEResnet68   | [4,4,3]                             | Lookahead+SGD   | 0.1  | False        | True          | 32         | 4.70M  | 91.82%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/without_aug_model)                                            |
| 7       | SEResnet68   | [4,4,3]                             | Lookahead+SGD   | 0.1  | True         | False         | 32         | 4.70M  | 95.80%   | [LINK](https://github.com/Mypainismorethanyours/SEResNet68/tree/main/Model_Weights_and_Eval_Metrics/without_gradient_model)                                      |


## 👩‍⚖️ Acknowledgement
*Authors:* 
