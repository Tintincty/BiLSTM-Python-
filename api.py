import json

import torch
from flask import Flask, request, render_template

from models.bilstm import BiLSTM
from utils import predict_transform_data
import os



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 获取模型名称
json_config = {}
with open("D:/PyTorch-RNN/config_file", 'r', encoding='utf-8') as f:
    json_config = json.load(f)
label_id2name = {'0': '负向情感', '1': '正向情感'}
print(json_config)
print(label_id2name)

model = BiLSTM(json_config['vocab_size'], json_config['embedding_dim'], json_config['hidden_size'],
               json_config['num_layers'], json_config['pad_idx'], json_config['unk_idx'])
model.load_state_dict(torch.load(json_config['ckpts'], map_location='cpu'))  # CPU模式下可独立运行
model.eval()


@app.route('/')
def hello_world():
    return render_template('index.htm')


@app.route('/sentiment', methods=['POST'])
def sentiment():
    sentence = request.form.get('text')  # "有朋友真好喔 朋友有我也真好四月快结束了 一切就都重新开始吧"
    print('input data = ', sentence)
    # 预测过程-梯度计算失效
    # https://pytorch.org/docs/master/autograd.html?highlight=no_grad#torch.autograd.no_grad
    with torch.no_grad():
        tag_to_id = json_config['tag_to_id']
        print('tag_to_id = ', tag_to_id)

        data, seq_len = predict_transform_data(sentence, json_config['word_to_id'], json_config['batch_size'])
        print('data.shape = ', data.shape)
        print('seq_len.shape = ', seq_len.shape)
        print('model feature input data = ', data)
        print('model text_lengths seq_len = ', seq_len)
        pred_prob = model(data.t(), seq_len)

        #  获取最大可能性索引id
        #  https://pytorch.org/docs/master/torch.html?highlight=argmax#torch.argmax
        # tag_to_id 通过torchtext 给出对应关系，与实际文本正好相反
        y_pred = pred_prob.argmax(dim=1).item()
        # 通过softmax 计算类别的可能性评分
        # https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.softmax
        probs = torch.nn.functional.softmax(pred_prob, dim=1)
        print(probs[0].numpy())  # tensor([[0.8472, 0.1528]])->[[0.8472015  0.15279852]]
        y_pred = dict([(v, k) for k, v in tag_to_id.items()])[y_pred]
        print(y_pred, label_id2name[str(y_pred)])  # 1 积极

        ## values() 是原始数据id列表
        ## probs[0].numpy() 预测的数据id列表
        id_prob_mapping = list(zip(tag_to_id.keys(), probs[0].numpy()))  # < 原始文本id，预测id〉对应关系
        results = [(id_prob[0], id_prob[1], label_id2name[str(id_prob[0])]) for id_prob in
                   id_prob_mapping]
        print(results)

        text_str = ""
        for tup in results:
            prob = tup[1]
            description = tup[2]
            text_str += "{}:{}\n".format(description, "{:.2f}%".format(prob * 100))
        print("text_str = ", text_str)
    return text_str


if __name__ == '__main__':

    app.run(host='127.0.0.1')
