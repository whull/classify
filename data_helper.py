# -*- coding: utf-8 -*-
"""
created on 20210119
@author: whull
意图识别 数据载入文件
"""

import csv

import numpy as np

import utils


def load_vocabulary(file):
    # file = os.path.join(path, 'dict', 'char_vocab.txt')
    fp = open(file, encoding='utf-8')
    lines = fp.readlines()
    fp.close()
    lines = [l.strip().split(',')[0] for l in lines]
    char2id = {char: i for i, char in enumerate(lines)}
    id2char = {i: char for i, char in enumerate(lines)}
    print(f"字典load完成，共{len(char2id)}个字符")
    return char2id, id2char


class LoadTrainData(object):
    def __init__(self, trainfile, hp):
        self.train_file = trainfile
        self.hp = hp
        self.start_id = 0

    def next_batch(self):
        train_data = []
        labels = []
        for exam in csv.reader(open(self.train_file, 'r', encoding='utf-8')):
            textid = eval(exam[0])
            train_data.append(textid+[0]*(30-len(textid)))
            labels.append(exam[1])

        train_data = np.array(train_data)
        labels = np.array(labels)

        numexam = len(train_data)
        randindex = np.random.permutation(range(numexam))
        num_epoch = 0

        while True:
            end_id = self.start_id + self.hp.batch_size
            if end_id >= numexam:
                end_id = end_id - numexam
                left = randindex[self.start_id:]
                right = randindex[:end_id]
                batchindex = np.append(left, right)
                self.start_id = end_id  # 反转

                num_epoch += 1
                print(f'已完成{num_epoch}个epoch')
                randindex = np.random.permutation(range(numexam))
            else:
                batchindex = randindex[self.start_id:end_id]
                self.start_id = end_id

            batch_data, batch_label = train_data[batchindex], labels[batchindex]
            yield batch_data, batch_label


def loadwordvec(file):
    pass


# 预处理训练数据
class CreatTrainData(object):
    def __init__(self, hp):
        self.hp = hp

    def buildtrain(self, source_file):
        utils.constructdict(source_file, hp.vocabfile)
        char2id, _ = load_vocabulary(self.hp.vocabfile)
        label2id, _ = load_vocabulary(self.hp.labelfile)
        utils.convertchar2id(source_file, hp.trainfile, char2id, label2id)

    def buildtest(self, source_file, dest_file):
        char2id, _ = load_vocabulary(self.hp.vocabfile)
        utils.convertchar2id(source_file, dest_file, char2id)


if __name__ == '__main__':
    from hyperparameters import CNNHyperParams
    hp = CNNHyperParams()
    data_util = CreatTrainData(hp)
    data_util.buildtrain(r'dataset/2.csv')

    # data_set = LoadTrainData(hp.trainfile, hp)
    # for batch_x, batch_y in data_set.next_batch():
    #     a, b = batch_x, batch_y


