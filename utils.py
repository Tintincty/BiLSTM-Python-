import json

import jieba
import torch

from configs import BasicConfigs

# 加载
jieba.load_userdict('dictionary/机构_学校.lex')
bc = BasicConfigs()
# 定义一个tokenizer，将句子拆分成单个词
def chi_tokenizer(sentence):
    return [word for word in jieba.cut(sentence)]


def save_config(config, config_file):
    """
    保持模型配置文件
    参数以json format 存储
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def predict_transform_data(sentence, word_to_id, batch_size):
    tokens = chi_tokenizer(sentence)
    res = []
    for token in tokens:
        res.append(word_to_id.get(token, 0))
    PAD_IX = [1] * (batch_size - len(res))
    data = torch.tensor(res + PAD_IX).unsqueeze(0)
    seq_len = torch.tensor([len(res + PAD_IX)])
    return data, seq_len


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