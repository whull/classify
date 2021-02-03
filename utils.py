# -*- coding: utf-8 -*-
"""
created on 20210119
@author: whull
意图识别 工具函数
"""

import os
import csv
import logging

import numpy as np


__all__ = ["constructdict"]


def constructdict(sourcefile, vocabfile):
    """数据预处理，用于生成字典
    :param sourcefile:  str or list; source files
    :param vocabfile: str; vocab file
    :return:
    """
    if isinstance(sourcefile, str):
        tdf = [sourcefile]
    elif isinstance(sourcefile, list):
        tdf = sourcefile
    else:
        raise TypeError("sourcefile must be a str or list")

    chardict = dict()
    for file in tdf:
        for example in csv.reader(open(file, 'r', encoding='utf-8')):
            sentence = example[0]
            for char in sentence:
                if char not in chardict:
                    chardict[char] = 1
                else:
                    chardict[char] += 1
    chars = sorted(chardict.items(), key=lambda a: a[1], reverse=True)  # 频次高的放前面节省内存
    writer = csv.writer(open(vocabfile, "w", encoding="utf-8", newline=""))
    pad = [("pad", 1), ("unk", 1)]
    pad.extend(chars)
    writer.writerows(pad)


def convertchar2id(file, destfile, char2id, label2id=None, EN=False):
    """汉字数据转index
    :param file:
    :param destfile:
    :param char2id:
    :param label2id:
    :return:
    """
    if label2id:
        domain = {i: 0 for i in label2id}
    writer = csv.writer(open(destfile, "w", encoding="utf-8", newline=""))
    for example in csv.reader(open(file, "r", encoding='utf-8')):
        sentence = example[0]
        if ' ' in sentence:
            continue

        if len(sentence) > 30:
            print(sentence)
            continue
        # 训练文本转换
        charids = []
        for char in sentence:
            charid = char2id[char] if char in char2id else char2id["unk"]
            charids.append(charid)

        # label 转换
        if label2id:
            label = example[-1]
            if label in label2id:
                gt = label2id[label]
                domain[label] += 1
            else:
                gt = label2id["other"]
                domain["other"] += 1
        else:
            gt = None

        writer.writerow((charids, gt))

    print(domain)

    print("数据转id处理完成")


def logger(save_file, level=logging.INFO, save_level=None):
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(save_file)
    if not save_level:
        save_level = level
    fh.setLevel(save_level)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


# @均值计算
class ValueWindow(object):
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


if __name__ == '__main__':
    path = r'dataset/2.csv'
    constructdict(path, 'dict/char_vocab.csv')
    # char2id, _ = load_vocabulary('dict/char_vocab.csv')
    # convertchar2id(path, r'dataset/train_id.csv', char2id)


