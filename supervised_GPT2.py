import numpy as np
import pandas as pd
import re
import h5py
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import log_loss, auc, roc_curve
from lxml import etree
from itertools import groupby
from gensim.models import Word2Vec
import glob
import math
import itertools
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
from tqdm import tqdm
# ##
# ## Load data
# ##
# print("Loading data...")
# xml_list = glob.glob('/content/*.xml')

# parser = etree.XMLParser(recover=True)

# def xml2df(xml_data):
#     root = etree.fromstring(xml_data, parser=parser) # element tree
#     all_records = []
#     for i, child in enumerate(root):
#         record = {}
#         for subchild in child:
#             record[subchild.tag] = subchild.text
#             all_records.append(record)
#     return pandas.DataFrame(all_records)

# dfs = []
# for ii in xml_list:
#     xml_data = open(ii, 'rb').read()
#     dfs.append(xml2df(xml_data))

# data = pandas.concat(dfs)
# data = data.drop_duplicates()
# data = data.sort_values("startDateTime")
# del dfs

# ##
# ## Create IP-dyad hours
# ##
# # print("De-dup Flows: "+str(len(data)))
# # data = data.sort_values('startDateTime')
# # data['totalBytes'] = data.totalSourceBytes.astype(float) + data.totalDestinationBytes.astype(float)
# # data['lowIP'] = data[['source','destination']].apply(lambda x: x[0] if x[0] <= x[1] else x[1], axis=1)
# # data['highIP'] = data[['source','destination']].apply(lambda x: x[0] if x[0] > x[1] else x[1], axis=1)
# # data['seqId'] = data['lowIP'] + '_' + data['highIP']  + '_' + data['startDateTime'].str[:13]
# # data['protoBytes'] = data[['protocolName','totalBytes']].apply(lambda x: str(x[0])[0] + str(math.floor(np.log2(x[1] + 1.0))), axis=1)
# print("De-dup Flows: "+str(len(data)))
# print("Creating undirected IP-dyads...")
# data['seqId'] = data['source'] + '_' + data['destination'] + '_' + data['startDateTime'].str[:13]
# data['lowPort'] = np.where(data.destinationPort <= data.sourcePort, data['destinationPort'], data['sourcePort'])

# ##
# ## Group by key and produce sequences
# ##
# #key = data.groupby('seqId')[['Tag','protoBytes']].agg({"Tag":lambda x: "%s" % ','.join([a for a in x]),"protoBytes":lambda x: "%s" % ','.join([str(a) for a in x])})
# key = data.groupby('seqId')[['Tag','lowPort']].agg({"Tag":lambda x: "%s" % ','.join([a for a in x]),"lowPort":lambda x: "%s" % ','.join([str(a) if int(a)<10000 else "10000" for a in x])})
# attacks = [a.split(",") for a in key.Tag.tolist()]
# sequences = [a.split(",") for a in key.lowPort.tolist()]

# unique_tokens = list(set([a for b in sequences for a in b]))
# le = LabelEncoder()
# le.fit(unique_tokens)
# sequences = [le.transform(s).tolist() for s in sequences]
# sequences = [[b+1 for b in a] for a in sequences]

# # Normalization (Minmax)
# # 提取唯一token
# unique_tokens = list(set([a for b in sequences for a in b]))

# # 对唯一token进行编码
# le = LabelEncoder()
# le.fit(unique_tokens)
# encoded_tokens = le.transform(unique_tokens)

# # 对编码后的token进行归一化
# scaler = MinMaxScaler(feature_range=(0,1))  # 假设我们想要将值归一化到0到1之间
# normalized_tokens = scaler.fit_transform(encoded_tokens.reshape(-1, 1)).flatten()

# # 创建编码到归一化值的映射
# token_to_normalized = dict(zip(le.classes_, normalized_tokens))

# # 应用映射到原始序列
# sequences = [[token_to_normalized[token] for token in seq] for seq in sequences]

# # Transform into csv:
# # value = []
# # is_anomaly = []
# # for sequence in sequences:
# #     for seq in sequence:
# #         value.append(seq)

# # Convert original text tags into integers
# for i in range(len(attacks)):
#     for j in range(len(attacks[i])):
#         if attacks[i][j] == 'Attack':
#             #print('Attack!')
#             attacks[i][j] = 1
#         elif attacks[i][j] == 'Normal':
#             attacks[i][j] = 0
#         else:
#             print('Not Attack/Normal error!')
# # timestamp = [i + 1 for i in range(len(value))]

# sequence_attack = zip(attacks, sequences)

# ##
# ## Produce sequences for modeling
# ##
# na_value = 0.
# seq_len = 10
# seq_index = []
# seq_x = []
# seq_y = []
# seq_attack = []
# for si, (sa, ss) in enumerate(sequence_attack):
#     prepend = [0.] * (seq_len)
#     seq =  prepend + ss
#     seqa = prepend + sa
#     for ii in range(seq_len, len(seq)):
#         subseq = seq[(ii-seq_len+1):(ii+1)]
#         subseqa = seqa[(ii-seq_len+1):(ii+1)]
#         is_anomaly = int(any(subseqa))
#         vex = []
#         for ee in subseq:
#             try:
#                 vex.append(ee)
#             except:
#                 vex.append(na_value)
#         seq_x.append(vex)
#         seq_y.append(is_anomaly)
# #         seq_index.append(si)
# #         seq_attack.append(seqa[ii])

# ##
# ## Initialize One-hot-encoder
# ##
# # ohe = OneHotEncoder(sparse=False)
# # ohe_y = ohe.fit_transform(np.asarray(seq_y).reshape(-1, 1))
# # X = np.array(seq_x)

# # Get frame df for input of CLSTM
# #frame = pandas.DataFrame(preprocessing.normalize(seq_x))
# frame = pandas.DataFrame(seq_x)
# frame = pandas.concat([frame, pandas.DataFrame(seq_y)], axis=1)

# # class BatchGenerator(object):
# #     def __init__(self, batch_size, x, y, ohe):
# #         self.batch_size = batch_size
# #         self.n_batches = int(math.floor(np.shape(x)[0] / batch_size))
# #         self.batch_index = [a * batch_size for a in range(0, self.n_batches)]
# #         self.x = x
# #         self.y = y
# #         self.ohe = ohe

# #     def __iter__(self):
# #         for bb in itertools.cycle(self.batch_index):
# #             y = self.y[bb:(bb+self.batch_size)]
# #             ohe_y = self.ohe.transform(y.reshape(len(y), 1))
# #             yield (self.x[bb:(bb+self.batch_size),], ohe_y)

# print("Ready to Go!")

frame = pd.read_csv('data.csv')

df = frame
n = len(df)
top_10_percent = df.head(n // 10)
df = top_10_percent

# 分离特征和标签
X = df.iloc[:, :-1].values  # 取前10列作为特征
y = df.iloc[:, -1].values   # 取最后一列作为标签

# 对特征进行标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将数据转换为GPT可以接受的文本格式
def convert_to_text(X_row):
    # 将每一行特征转换为文本格式
    return ' '.join([f"feature_{i}: {value}" for i, value in enumerate(X_row)])

# 将每一行数据转换为文本
X_text = [convert_to_text(row) for row in X]

# 将标签转换为 Tensor
y_tensor = torch.tensor(y, dtype=torch.long)

# 创建Dataset和DataLoader
class TabularTextDataset(Dataset):
    def __init__(self, X_text, y):
        self.X_text = X_text
        self.y = y

    def __len__(self):
        return len(self.X_text)

    def __getitem__(self, idx):
        return self.X_text[idx], self.y[idx]

# 将数据集划分为训练集和测试集
dataset = TabularTextDataset(X_text, y_tensor)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 加载GPT-2模型和Tokenizer
model_name = "gpt2"  # 使用GPT-2作为示例
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 解决pad_token问题：将pad_token设置为eos_token
tokenizer.pad_token = tokenizer.eos_token  # 或者使用 tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 加载GPT-2的分类头模型
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

model.config.pad_token_id = model.config.eos_token_id

# 调整模型的输入处理
model.resize_token_embeddings(len(tokenizer))

# 将模型移动到GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 损失函数和优化器
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=5e-6)

# 存储训练过程中的评估指标
train_losses = []
train_accuracies = []
train_precisions = []
train_recalls = []
train_f1s = []
train_aucs = []

test_accuracies = []
test_precisions = []
test_recalls = []
test_f1s = []
test_aucs = []

# 训练模型
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Training loop
    for inputs, labels in tqdm(train_loader):
        # 对文本进行tokenization，加入padding和truncation
        inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 获取模型的输出
        outputs = model(**inputs)
        logits = outputs.logits

        # 计算损失
        loss = criterion(logits.view(-1, 2), labels.view(-1))
        loss.backward()
        optimizer.step()

        # 累积损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 保存预测值和真实标签
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算训练集的评估指标
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds)
    epoch_recall = recall_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_preds)

    # 存储训练过程中的评估指标
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    train_precisions.append(epoch_precision)
    train_recalls.append(epoch_recall)
    train_f1s.append(epoch_f1)
    train_aucs.append(epoch_auc)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, "
          f"Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1-Score: {epoch_f1:.4f}, AUC: {epoch_auc:.4f}")

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 对文本进行tokenization，加入padding和truncation
            inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
            labels = labels.to(device)

            # 获取模型的输出
            outputs = model(**inputs)
            logits = outputs.logits

            # 保存预测值和真实标签
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算测试集的评估指标
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)
    test_auc = roc_auc_score(all_labels, all_preds)

    # 存储测试过程中的评估指标
    test_accuracies.append(test_acc)
    test_precisions.append(test_precision)
    test_recalls.append(test_recall)
    test_f1s.append(test_f1)
    test_aucs.append(test_auc)

    print(f"Test Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, "
          f"F1-Score: {test_f1:.4f}, AUC: {test_auc:.4f}")

# 绘制评估指标图表
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 绘制训练和测试集的指标
axes[0, 0].plot(range(1, epochs+1), train_accuracies, label='Train Accuracy', color='blue')
axes[0, 0].plot(range(1, epochs+1), test_accuracies, label='Test Accuracy', color='red')
axes[0, 0].set_title('Accuracy Over Epochs')
axes[0, 0].legend()

axes[0, 1].plot(range(1, epochs+1), train_precisions, label='Train Precision', color='blue')
axes[0, 1].plot(range(1, epochs+1), test_precisions, label='Test Precision', color='red')
axes[0, 1].set_title('Precision Over Epochs')
axes[0, 1].legend()

axes[0, 2].plot(range(1, epochs+1), train_recalls, label='Train Recall', color='blue')
axes[0, 2].plot(range(1, epochs+1), test_recalls, label='Test Recall', color='red')
axes[0, 2].set_title('Recall Over Epochs')
axes[0, 2].legend()

axes[1, 0].plot(range(1, epochs+1), train_f1s, label='Train F1-Score', color='blue')
axes[1, 0].plot(range(1, epochs+1), test_f1s, label='Test F1-Score', color='red')
axes[1, 0].set_title('F1-Score Over Epochs')
axes[1, 0].legend()

axes[1, 1].plot(range(1, epochs+1), train_aucs, label='Train AUC', color='blue')
axes[1, 1].plot(range(1, epochs+1), test_aucs, label='Test AUC', color='red')
axes[1, 1].set_title('AUC Over Epochs')
axes[1, 1].legend()

plt.tight_layout()
plt.show()