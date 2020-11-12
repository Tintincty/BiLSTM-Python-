"""
构建vocabulary
#
"""

# 导入torch库
import torch
# 导入torchtext，主要用于自然语言处理预处理
from torchtext.data import Field, TabularDataset
# 加载外部词向量
from torchtext.vocab import Vectors
# 引入自定义配置
from configs import BasicConfigs
# 引入工具类
from utils import chi_tokenizer

config = BasicConfigs()


def load_dataset():
    """
    定义数据加载方法
    :return:
    """
    # 定义字段 (TEXT/LABEL)
    # include_lengths=True 为了方便后续使用torch pack_padded_sequence
    # chi_tokenizer 分词器，主要对我们的每个句子进行切分
    TEXT = Field(tokenize=chi_tokenizer, include_lengths=True)
    LABEL = Field(eos_token=None, pad_token=None, unk_token=None)
    # pad参数用来补充embedding
    # torchtext中于文件配对关系
    fields = [('data', TEXT), ('label', LABEL)]
    # 加载数据
    ## 注意skip_header = True 加载数据
    train_data, val_data = TabularDataset.splits(path='data',
                                                 train='train.csv',
                                                 validation='val.csv',
                                                 format='csv',
                                                 fields=fields,
                                                 skip_header=True)

    ## 数据记录数统计
    print('train total_count = ', len(train_data.examples))
    print('val total_count = ', len(val_data.examples))

    # 构建字典

    ## 构建从本地加载的词向量
    vectors = Vectors(name=config.embedding_loc, cache=config.cach)
    ## 构建词汇
    TEXT.build_vocab(train_data, max_size=25000, vectors=vectors, unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data, vectors=vectors)

    print(f'Unique tokens in TEXT vocabulary:{len(TEXT.vocab)}')

    print(f'Unique tokens in LABEL vocabulary:{len(LABEL.vocab)}')

    return train_data, val_data, TEXT, LABEL


if __name__ == '__main__':
    load_dataset()
