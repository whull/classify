# -*- coding: utf-8 -*-
"""
created on 20210119
@author: whull
意图识别 网络类
"""

import tensorflow as tf

import utils
import data_helper


class CNNClassify(object):

    def __init__(self, hp, is_training=False):
        self.hp = hp
        self.is_training = is_training

    def build_test_model(self, ):
        self.x = tf.placeholder(tf.int32, shape=[None, None])
        logits = self.cnnmodel(self.x)
        self.prob = tf.nn.softmax(logits)
        self.y_pred_cls = tf.argmax(self.prob, axis=-1, output_type=tf.int32)  # 预测类别

    def build_train_model(self,):
        self.x = tf.placeholder(tf.int32, shape=[None, None], name="input_x")
        self.y = tf.placeholder(tf.int32, shape=[None], name="input_y")
        logits = self.cnnmodel(self.x)
        self.loss, self.acc = self.loss_layer(logits, self.y)

        self.global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = self.hp.learning_rate * utils.decay_lr(self.hp.warmup_steps, self.global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hp.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("acc", self.acc)
        self.summary_op = tf.summary.merge_all()

    def loss_layer(self, logits, Y):
        # Y = tf.expand_dims(Y, axis=-1)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        prob = tf.nn.softmax(logits)
        y_pred_cls = tf.argmax(prob, axis=-1, output_type=tf.int32)  # 预测类别
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(Y, y_pred_cls)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return loss, acc

    def cnnmodel(self, inputs):
        char2id, _ = data_helper.load_vocabulary(self.hp.vocabfile)
        vocabsize = len(char2id)
        with tf.variable_scope("embedding_layer"):
            embedding = tf.get_variable("embedding_vocab", shape=[vocabsize, self.hp.embedding_size])
            # load 预训练字向量
            if self.hp.use_pre_vec:
                pre_embedding = data_helper.loadwordvec(self.hp.wordvecfile)
                embedding = tf.assign(embedding, pre_embedding)
            embedding_inputs = tf.nn.embedding_lookup(embedding, inputs)
        # n*len*1*d
        layer_inputs = tf.expand_dims(embedding_inputs, axis=2)
        for i in range(self.hp.num_blocks):
            with tf.variable_scope(f"block_{i}"):
                cnnoutputs = []
                for size in self.hp.filtersizes.split(','):
                    with tf.variable_scope(f'filter_{size}'):
                        # tf.layers.conv2d() 简单版
                        weight = tf.get_variable("weight", shape=[int(size), 1, self.hp.embedding_size, self.hp.embedding_size])
                        cnnoutput = tf.nn.conv2d(layer_inputs, weight, [1, 1, 1, 1], padding="SAME")
                        cnnoutputs.append(cnnoutput)
                layer_outputs = tf.concat(cnnoutputs, axis=-1)
                layer_inputs = tf.layers.dense(layer_outputs, self.hp.embedding_size)

        layer_outputs = tf.reduce_max(tf.squeeze(layer_inputs, axis=2), axis=1)
        fc = tf.layers.dense(layer_outputs, self.hp.num_units, name="fc")
        if self.is_training:
            fc = tf.nn.dropout(fc, 0.5)
        fc = tf.nn.relu(fc)

        logits = tf.layers.dense(fc, self.hp.num_classes, name="output_layer")

        return logits





