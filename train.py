# -*- coding: utf-8 -*-
"""
created on 20210121
@author: whull
意图识别 训练代码
"""

import os
import time

import tensorflow as tf

import utils
import data_helper
from network import CNNClassify
from hyperparameters import CNNHyperParams


def main(_):
    hp = CNNHyperParams()
    log = utils.logger(hp.logfile, level=tf.logging.INFO)
    data_set = data_helper.LoadTrainData(hp.trainfile, hp)
    # {'other': 7263, 'music': 9501, 'movie': 12816, 'time': 133}

    model = CNNClassify(hp, is_training=True)
    model.build_train_model()

    time_window = utils.ValueWindow(100)
    loss_window = utils.ValueWindow(100)
    acc_window = utils.ValueWindow(100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(hp.logdir)
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=6)
        checkpoint_path = os.path.join(hp.logdir, 'model.ckpt')
        restore_path = tf.train.latest_checkpoint(hp.logdir)
        if restore_path is None:
            log.info('Starting new training')
        else:
            saver.restore(sess, restore_path)
            log.info('Resuming from checkpoint: %s' % restore_path)

        for batch_x, batch_y in data_set.next_batch():
            start_time = time.time()
            loss, acc, step, _summary, _ = sess.run(
                [model.loss, model.acc, model.global_step, model.summary_op, model.train_op],
                feed_dict={model.x: batch_x, model.y: batch_y})

            time_window.append(time.time() - start_time)
            loss_window.append(loss)
            acc_window.append(acc)

            if step % 50 == 0:
                message = 'Step %-7d [%.03f sec/step, avg_loss=%.05f, avg_acc=%.05f]' % (
                    step, time_window.average, loss_window.average, acc_window.average)
                log.info(message)
                summary_writer.add_summary(_summary, step)
            if step % 500 == 0:
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    tf.app.run()
