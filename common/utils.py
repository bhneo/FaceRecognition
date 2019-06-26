import callbacks

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import backend as K


def get_lr_schedule(steps, decay=0.1):
    def lr_schedule(step, lr):
        for i in steps:
            if step == i:
                lr *= decay
        return lr
    return lr_schedule


def update_learning_rate(model, steps, step):
    if not hasattr(model.optimizer, 'lr'):
        raise ValueError('Optimizer must have a "lr" attribute.')
    last_lr = float(K.get_value(model.optimizer.lr))
    lr = last_lr
    for i in range(len(steps)):
        if step < steps[i][0]:
            break
        lr = steps[i][1]
    if last_lr != lr:
        K.set_value(model.optimizer.lr, lr)
        print('\nStep %05d: LearningRateScheduler reducing learning '
              'rate to %s.' % (step, lr))


def write_log(tensor_board, names, logs, step):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        tensor_board.writer.add_summary(summary, step)
        tensor_board.writer.flush()

