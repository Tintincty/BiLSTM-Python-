'''

Some helper functions for PyTorch, including:
'''
import os

import torch

__all__ = ['save_checkpoint']

def save_checkpoint(state, is_best, checkpoint='ckpts', filename='ckpts.pth.tar'):

    """
    模型保存方法
    :param state:
    :param is_best:
    :param checkpoint:
    :param filename:
    :return:
    """
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    # 保存断点信息
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # 模型保存
    if is_best:
        model_name = 'bilstm_sentiment.pth'
        model_path = os.path.join(checkpoint, model_name)
        torch.save(state['state_dict'], model_path)
