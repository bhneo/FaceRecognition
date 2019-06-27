import tensorflow as tf
import numpy as np
import verification

import sklearn
import data_input
from tensorflow import keras
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback


class LearningRateSchedulerOnBatch(Callback):
    def __init__(self, schedule, steps_per_epoch, verbose=0):
        super(LearningRateSchedulerOnBatch, self).__init__()
        self.schedule = schedule
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self.epoch = 0

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        if self.verbose:
            print('\nStart at learning rate of %s.' % (logs['lr']))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        global_step = self.epoch * self.steps_per_epoch + batch
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        last_lr = float(K.get_value(self.model.optimizer.lr))
        if global_step % 1000 == 0:
            print('lr-batch-epoch:', last_lr, batch, self.epoch)
        lr = self.schedule(batch, last_lr)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if last_lr != lr:
            K.set_value(self.model.optimizer.lr, lr)
            logs = logs or {}
            logs['lr'] = K.get_value(self.model.optimizer.lr)
            if self.verbose > 0:
                print('\nStep %05d: LearningRateScheduler reducing learning '
                      'rate to %s.' % (global_step + 1, lr))


class FaceRecognitionValidation(Callback):
    def __init__(self, extractor, steps_per_epoch, valid_list, period, verbose=0):
        super(FaceRecognitionValidation, self).__init__()
        self.extractor = extractor
        self.steps_per_epoch = steps_per_epoch
        self.valid_list = valid_list
        self.period = period
        self.verbose = verbose
        self.global_step = 0
        self.save_step = 0
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        global_step = self.epoch * self.steps_per_epoch + batch
        if global_step >= 0 and global_step % self.period == 0:
            acc_list = []
            logs = logs or {}
            for key in self.valid_list:
                print('Test on valid set:', key)
                bins, is_same_list = self.valid_list[key]
                dataset = tf.data.Dataset.from_tensor_slices(bins) \
                    .map(data_input.get_valid_parse_function(False), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                    .batch(256)
                dataset_flip = tf.data.Dataset.from_tensor_slices(bins) \
                    .map(data_input.get_valid_parse_function(True), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                    .batch(256)
                batch_num = len(bins) // 256
                if len(bins) % 256 != 0:
                    batch_num += 1
                embeddings = self.extractor.predict(dataset, steps=batch_num, verbose=1)
                embeddings_flip = self.extractor.predict(dataset_flip, steps=batch_num, verbose=1)
                embeddings_parts = [embeddings, embeddings_flip]
                x_norm = 0.0
                x_norm_cnt = 0
                for part in embeddings_parts:
                    for i in range(part.shape[0]):
                        embedding = part[i]
                        norm = np.linalg.norm(embedding)
                        x_norm += norm
                        x_norm_cnt += 1
                x_norm /= x_norm_cnt
                embeddings = embeddings_parts[0] + embeddings_parts[1]
                embeddings = sklearn.preprocessing.normalize(embeddings)
                print(embeddings.shape)
                _, _, accuracy, val, val_std, far = verification.evaluate(embeddings, is_same_list, folds=10)
                acc, std = np.mean(accuracy), np.std(accuracy)

                print('[%s][%d]XNorm: %f' % (key, batch, x_norm))
                print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (key, batch, acc, std))
                acc_list.append(acc)

            logs['score'] = sum(acc_list)
            if self.verbose > 0:
                print('\nScore of step %05d: %0.5f' % (global_step, logs['score']))


class ModelCheckpointOnBatch(Callback):
    def __init__(self,
                 filepath,
                 steps_per_epoch,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1):
        super(ModelCheckpointOnBatch, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.steps_per_epoch = steps_per_epoch
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epoch = 0

        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        global_step = self.epoch * self.steps_per_epoch + batch
        logs = logs or {}
        if global_step % self.period == 0:
            filepath = self.filepath.format(step=batch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nStep %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (global_step + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nStep %05d: %s did not improve from %0.5f' %
                                  (global_step + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nStep %05d: saving model to %s' % (global_step + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


