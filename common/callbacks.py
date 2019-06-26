import tensorflow as tf
import numpy as np
import verification

from tensorflow import keras
from tensorflow.python.keras.callbacks import Callback


class FaceRecognitionTest(Callback):
    def __init__(self, extractor, valid_list, valid_name_list, period, batch_size, verbose=0):
        super(FaceRecognitionTest, self).__init__()
        self.extractor = extractor
        self.valid_list = valid_list
        self.valid_name_list = valid_name_list
        self.period = period
        self.batch_size = batch_size
        self.verbose = verbose
        self.highest_acc = [0.0, 0.0]  # lfw and target
        self.global_step = 0
        self.save_step = 0

    def on_batch_end(self, batch, logs=None):
        if batch >= 0 and batch % self.period == 0:
            acc_list = []
            for i in range(len(self.valid_list)):
                embeddings = test(self.valid_list[i], self.extractor, self.batch_size, 10)

                # get embedding
                _, _, accuracy, val, val_std, far = verification.evaluate(embeddings, self.valid_name_list, nrof_folds=fold)
                acc, std = np.mean(accuracy), np.std(accuracy)
                return acc, std, _xnorm, embeddings_list

                print('[%s][%d]XNorm: %f' % (self.valid_name_list[i], self.batch_size, x_norm))
                print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.valid_name_list[i], self.batch_size, acc, std))
                acc_list.append(acc)

            self.save_step += 1

            is_highest = False
            if len(acc_list) > 0:
                score = sum(acc_list)
                if acc_list[-1] >= self.highest_acc[-1]:
                    if acc_list[-1] > self.highest_acc[-1]:
                        is_highest = True
                    else:
                        if score >= self.highest_acc[0]:
                            is_highest = True
                            self.highest_acc[0] = score
                    self.highest_acc[-1] = acc_list[-1]
            if is_highest:
                print('saving', self.save_step)
                filepath = self.filepath.format(epoch=epoch + 1, **logs)
                self.model.save_weights(filepath, overwrite=True)
            print('[%d]Accuracy-Highest: %1.5f' % (mbatch, self.highest_acc[-1]))


class FaceRecognitionCheckpoint(Callback):
    def __init__(self,
                 ckpt_path,
                 extractor,
                 valid_list,
                 valid_name_list,
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 period=1):
        super(FaceRecognitionCheckpoint, self).__init__()
        self.extractor = extractor
        self.valid_list = valid_list
        self.valid_name_list = valid_name_list
        self.verbose = verbose
        self.ckpt_path = ckpt_path
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        self.monitor_op = np.greater
        self.best = -np.Inf

        self.highest_acc = [0.0, 0.0]  # lfw and target
        self.global_step = 0
        self.save_step = 0

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            ckpt_path = self.ckpt_path.format(step=batch + 1, **logs)
            if self.save_best_only:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nBatch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s' % (batch + 1, self.monitor, self.best,
                                                       current, ckpt_path))
                    self.best = current
                    if self.save_weights_only:
                        self.model.save_weights(ckpt_path, overwrite=True)
                    else:
                        self.model.save(ckpt_path, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nBatch %05d: %s did not improve from %0.5f' %
                              (batch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nBatch %05d: saving model to %s' % (batch + 1, ckpt_path))
                if self.save_weights_only:
                    self.model.save_weights(ckpt_path, overwrite=True)
                else:
                    self.model.save(ckpt_path, overwrite=True)


