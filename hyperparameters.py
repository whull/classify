# -*- coding: utf-8 -*-
"""
created on 20210119
@author: whull
意图识别 参数配置
"""


class CNNHyperParams(object):
    # 文件路径
    trainfile = r"dataset/train_id.csv"
    vocabfile = r"dict/char_vocab.csv"
    labelfile = r'dict/label_vocab.csv'

    use_pre_vec = False
    wordvecfile = r"dict/char_vector.csv"

    # 模型参数
    num_blocks = 2
    num_units = 512
    embedding_size = 100
    filtersizes = "2,3,4"
    num_classes = 4

    # 训练参数
    logdir = r"checkpoint"
    logfile = r"log/train.log"
    learning_rate = 0.001
    batch_size = 64


class MLPHyperParams(object):
    pass

