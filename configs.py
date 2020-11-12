import torch


class BasicConfigs():
    # parmeters for wordvector
    embedding_loc = 'embeddings/fasttext/cc.zh.300.vec'
    cach = '.vector_cache'  # 词向量的缓存位置
    # parameters for overall model training
   # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    dropout_rate = 0.1
    train_embedding = True
    batch_size = 32   #
    # paramaeter for birnn
    hidden_size = 50 # 100
    num_layers = 2   # 1

    # model 保存位置
    ckpts = "ckpts/bilstm_sentiment.pth"




if __name__ == '__main__':
    config = BasicConfigs()


    config_key = list(filter(lambda  x: not x.startswith('__'),dir(config)))

    kv_config = dict([(key,config.__getattribute__(key)) for key in config_key])

    for k,v in kv_config.items():
        print(k,v)
