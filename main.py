import argparse
import time
from collections import OrderedDict

# torch 基础库
import torch
from torch import nn
from torch import optim
# torchtext 专门用于文本处理
from torchtext.data import BucketIterator

from configs import BasicConfigs
# 定义工具
from dataset import load_dataset
from misc import save_checkpoint
from models.bilstm import BiLSTM
from utils import save_config

config = BasicConfigs()
device = config.device
print('PyTorch Training Sentiment device = ', device)

# 获取参数 定义轮数，学习率
parser = argparse.ArgumentParser()
parser.add_argument('--epoches', default=4, type=int, help='training iter count')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

args = parser.parse_args()


def config_model(vocab_size, embedding_dim, pad_idx, unk_idx, word_to_id, tag_to_id):
    """

    :param vocab_size: 词典大小
    :param embedding_dim: 每个单词embeddding 维度大小(FastText 外部embedding）
    :param pad_idx:
    :param unk_idx:
    :param word_to_id: 字典
    :param tag_to_id:标签
    :return:
    """

    mappings = OrderedDict()

    # BasicConfigs 下的参数
    config_key = list(filter(lambda x: not x.startswith('__'), dir(config)))
    kv_config = dict([(key, config.__getattribute__(key)) for key in config_key])
    print('kv_config = ', kv_config)

    for k, v in kv_config.items():
        mappings[k] = v

    # 程序动态参数
    mappings['vocab_size'] = vocab_size
    mappings['embedding_dim'] = embedding_dim
    mappings['pad_idx'] = pad_idx
    mappings['unk_idx'] = unk_idx
    mappings['word_to_id'] = word_to_id
    mappings['tag_to_id'] = tag_to_id

    # model path
    return mappings


# train
def train(model, train_iter, optimizer, loss_func):
    """
    :param model:
    :param train_iter:
    :param optimizer:
    :param loss_func:
    :return:
    """

    ### 设置model模式
    model.train()

    total_train_loss = 0.
    total_train_acc = 0.
    step = 1
    for iter_num, batch in enumerate(train_iter):
        # 获取X和y的数据
        X, text_lengths = batch.data   # text,text_lengths
        X, text_lengths = X.to(device), text_lengths.to(device)
        y = batch.label.squeeze(0).to(device)

        score = model(X, text_lengths)
        loss = loss_func(score, y)
        acc = (score.argmax(dim=1) == y).sum().cpu().item() / len(y)
        total_train_acc += acc
        total_train_loss += loss

        step += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 平均数据
    avg_train_acc = total_train_acc / step
    avg_train_loss = total_train_loss / step

    return avg_train_acc, avg_train_loss


# val
def val(model, val_iter, loss_func):
    """

    :param model:
    :param val_iter:
    :param loss_func:
    :return:
    """
    ### 设置model模式
    model.eval()

    total_val_loss = 0.
    total_val_acc = 0.
    step = 1

    with torch.no_grad():
        for iter_num, batch in enumerate(val_iter):
            # 获取X和y的数据
            X, text_lengths = batch.data  # text,text_lengths
            X, text_lengths = X.to(device), text_lengths.to(device)
            y = batch.label.squeeze(0).to(device)

            score = model(X, text_lengths)
            loss = loss_func(score, y)
            acc = (score.argmax(dim=1) == y).sum().cpu().item() / len(y)
            total_val_acc += acc
            total_val_loss += loss

            step += 1

    # 平均数据
    avg_val_acc = total_val_acc / step
    avg_val_loss = total_val_loss / step

    return avg_val_acc, avg_val_loss


# 定义入口程序
if __name__ == '__main__':
    # 数据处理
    ## 加载数据
    train_data, val_data, TEXT, LABEL = load_dataset()
    ## 封装数据
    train_iter = BucketIterator(train_data,
                                batch_size=config.batch_size,
                                sort_key=lambda x: len(x.data),
                                sort_within_batch=True,
                                shuffle=True,
                                device=device)

    val_iter = BucketIterator(val_data,
                              batch_size=config.batch_size,
                              sort_key=lambda x: len(x.data),
                              sort_within_batch=True,
                              shuffle=False,
                              device=device)

    vocab_size = TEXT.vocab.vectors.shape[0]
    embedding_dim = TEXT.vocab.vectors.shape[1]
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]

    print('pad_idx = ', pad_idx)
    print('unk_idx = ', unk_idx)
    print('vocab_size = ', vocab_size)
    print('embedding_dim = ', embedding_dim)
    # 打印上述四句话的参数信息

    ## 保存参数
    word_to_id = dict(TEXT.vocab.stoi)
    tag_to_id = dict(LABEL.vocab.stoi)
    param_config = config_model(vocab_size, embedding_dim, pad_idx, unk_idx, word_to_id, tag_to_id)
    save_config(param_config, 'config_file')
    # 初始化模型
    pre_trained_embedding = TEXT.vocab.vectors
    model = BiLSTM(vocab_size, embedding_dim, config.hidden_size,
                   config.num_layers, pad_idx, unk_idx, pre_trained_embedding=pre_trained_embedding)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    # 模型训练
    ## train/val/保存最优模型
    print('=>model training ...<=')
    best_val_loss = float('inf')
    N_EPOCH = args.epoches
    for epoch in range(N_EPOCH):
        t1 = time.time()

        # train
        train_acc, train_loss = train(model, train_iter, optimizer, loss_func)
        # val
        val_acc, val_loss = val(model, val_iter, loss_func)

        diff = (time.time() - t1)
        print("Epoch [{}/{}] Train acc {:.4f} Train loss {:.4f} "
              "Val acc {:.4f} Val loss {:.4f} Time:{}".format(epoch + 1, N_EPOCH,
                                                              train_acc, train_loss,
                                                              val_acc, val_loss,
                                                              int(diff)))

        # 保存最优模型
        if val_loss < best_val_loss:
            is_best = True
            print('save model loss descreasing {:.4f}->{:.4f}'.format(best_val_loss, val_loss))
            best_val_loss = val_loss

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': best_val_loss,
                'optimizer': optimizer.state_dict()
            }, is_best)
