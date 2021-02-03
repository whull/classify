# -*- coding: utf-8 -*-
"""
created on 20210121
@author: whull
意图识别 训练代码
"""

import numpy as np
import tensorflow as tf

import data_helper
from network import CNNClassify
from hyperparameters import CNNHyperParams


hp = CNNHyperParams()
model = CNNClassify(hp, is_training=False)
model.build_test_model()
char2id, _ = data_helper.load_vocabulary(hp.vocabfile)
_, id2label = data_helper.load_vocabulary(hp.labelfile)

with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=6)
    restore_path = tf.train.latest_checkpoint(hp.logdir)
    saver.restore(sess, restore_path)
    while True:
        x = input("请输入内容：")
        input_id = np.zeros([1, 30])
        x_id = [char2id[i] if i in char2id else char2id["unk"] for i in x]
        input_id[0, 0:len(x_id)] = x_id
        pred_id = sess.run(model.y_pred_cls, feed_dict={model.x: input_id})
        print(pred_id, id2label[pred_id[0]])

