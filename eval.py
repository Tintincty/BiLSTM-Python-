import pandas as pd
from models.bilstm import BiLSTM
import json
import torch
import numpy as np
from sklearn import metrics


# 获取模型名称
json_config = {}
with open("config_file", 'r', encoding='utf-8') as f:
    json_config = json.load(f)
label_id2name = {'0': '负向情感', '1': '正向情感'}
#print(json_config['word_to_id'])
print(label_id2name)
tag_to_id = json_config['tag_to_id']## 注意： {'1': 0, '0': 1} 这里的key对应原始文本数据，value是torchtext建立的映射关系
id_to_tag = dict([(v,k) for k,v in tag_to_id.items()])
print('tag_to_id=',tag_to_id) # k 原始文本数据
print('id_to_tag=',id_to_tag) # k 为torchtext加载后的key

model = BiLSTM(json_config['vocab_size'], json_config['embedding_dim'], json_config['hidden_size'],
               json_config['num_layers'], json_config['pad_idx'], json_config['unk_idx'])
model.load_state_dict(torch.load(json_config['ckpts'], map_location='cpu'))  # CPU模式下可独立运行
model.eval()

import jieba
# 定义一个tokenizer
def chi_tokenizer(sentence):
    return [word for word in jieba.cut(sentence)]

def transform_data(record, word_to_id, tag_to_id, batch_size):
    tokens = chi_tokenizer(record['data'])
    res = []
    for token in tokens:
        res.append(word_to_id.get(token, 0))
    PAD_IX = [1] * (batch_size - len(res))

    #  句子专为word_to_id 数据
    data = torch.tensor(res + PAD_IX).unsqueeze(0)

    # 句子padding_index长度
    seq_len = torch.tensor([len(res + PAD_IX)])

    # 句子中label tag_to_id 数据
    # record['label']: 原始文本的id
    label = torch.tensor(record['label'])
    return data, seq_len, label
test_data = pd.read_csv('data/test.csv',index_col=0)
test_data.head()
print(test_data.shape)
df_len = test_data.shape[0]
print('test rows = {}'.format(df_len))

predict_all = np.array([], dtype=int)
labels_all = np.array([], dtype=int)
for i in range(df_len):
    # 获取每行数据所有列
    # 专为dict对象
    record = test_data.loc[i, :].to_dict()
    data, seq_len, label = transform_data(record, json_config['word_to_id'],
                                          json_config['tag_to_id'],
                                          json_config['batch_size'])
    pred_prob = model(data.t(), seq_len)
    y_pred = pred_prob.argmax(dim=1).item()  # torchtext 标签id对应关系同原始数据反过来的
    y_pred = int(id_to_tag[y_pred])  # 预测结果id专为实际文本的id，str类型

    labels_all = np.append(labels_all, int(label.item()))  # 数据赋值
    predict_all = np.append(predict_all, int(y_pred))

acc = metrics.accuracy_score(labels_all, predict_all)
report = metrics.classification_report(labels_all, predict_all, target_names=label_id2name.values(), digits=4)
confusion_matrix = metrics.confusion_matrix(labels_all, predict_all)
print("Precision, Recall and F1-Score...")
print(report)
print("Confusion Matrix...")
print(confusion_matrix)