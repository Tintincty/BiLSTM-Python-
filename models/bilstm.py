# 导入定义bilstm库
import torch
from torch import nn


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, pad_idx,
                 unk_idx, pre_trained_embedding=None):
        super(BiLSTM, self).__init__()

        # lookup table that stores embeddings of fixed dictionary and size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # 加载 预处理embeddding
        if pre_trained_embedding is not None:
            self.embedding.weight.data.copy_(pre_trained_embedding)

            # 对于pre_trained vocab 没有对应的vector
            self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
            self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

        # 定义encoder
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=True)

        # 定义decoder
        self.decoder = nn.Linear(2 * hidden_size, 2)

        # 定义dropout
        ## 减少过拟合，增加我们模型的泛化能力
        ## Dropout 在深度网络训练中，以一定随机概率临时丢失一部分神经元的节点
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, text_lengths):
        """

        :param inputs:  每个batch text 内容
        :param text_lengths:  words 的长度,词汇长度
        :return:
        """

        # 提取词特征（词数，批量大小）
        embedding = self.dropout(self.embedding(inputs))

        # pad sequence <pad> 设置
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedding, text_lengths)

        # encoder
        ## 返回的h_last,c_last 就是移除padding字符后的hidden_state 和 cell_state
        ## 则LSTM 只会作用到实际长度的句子，而不会用无用padding
        ## 返回output仍然是PackSequence 类型
        packed_output, (hidden, cell) = self.encoder(packed_embedded)

        # decoder
        ## concat the final forward (hidden[-2,:,:]) and backward (hidden[-1:,:,:]) hidden layers
        ## and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        outs = self.decoder(hidden)
        return outs
