import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import multi_gpu_model

import data_input
from nets import fmobilefacenet
from common import block, utils, callbacks
from config import config, default, generate_config
from losses.face_losses import MarginSoftmaxSparseCategoricalCrossentropy


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--dataset', default=default.dataset, help='dataset config')
    parser.add_argument('--network', default=default.network, help='network config')
    parser.add_argument('--loss', default=default.loss, help='loss config')
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset, args.loss)
    parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
    parser.add_argument('--pretrained', default=default.pretrained, help='pretrained model to load')
    parser.add_argument('--pretrained-epoch', type=int, default=default.pretrained_epoch, help='pretrained epoch to load')
    parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')
    parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=default.wd, help='weight decay')
    parser.add_argument('--mom', type=float, default=default.mom, help='momentum')
    parser.add_argument('--frequent', type=int, default=default.frequent, help='')
    parser.add_argument('--kvstore', type=str, default=default.kvstore, help='kvstore setting')
    args = parser.parse_args()
    return args


def build_model(input_shape, args):
    data = keras.Input(shape=input_shape, name='data')
    embedding = eval(config.net_name).get_symbol(data, config.emb_size, None, config.net_act, args.wd)
    extractor = keras.Model(inputs=data, outputs=embedding, name='extractor')

    fc7 = block.FaceCategoryLogits(config.num_classes, norm=True)(embedding)
    classifier = keras.Model(inputs=data, outputs=fc7, name='classifier')

    return extractor, classifier


def train_net(args):
    data_dir = config.dataset_path
    image_size = config.image_shape[0:2]
    assert len(image_size) == 2
    assert image_size[0] == image_size[1]
    print('image_size', image_size)
    print('num_classes', config.num_classes)
    training_path = os.path.join(data_dir, "train.tfrecords")
    print('Called with argument:', args, config)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices:{}'.format(strategy.num_replicas_in_sync))
    default.batch_size = default.per_batch_size * strategy.num_replicas_in_sync
    print('batch size:{}'.format(default.batch_size))
    train_dataset, batches_per_epoch = data_input.training_dataset(training_path, default.batch_size)

    extractor, classifier = build_model((image_size[0], image_size[1], 3), args)
    if strategy.num_replicas_in_sync > 1:
        classifier = multi_gpu_model(classifier, strategy.num_replicas_in_sync)

    initial_step = 0
    load_path = None
    ckpt_path = os.path.join(args.models_root, '%s-%s-%s' % (args.network, args.loss, args.dataset), 'model-{step:04d}.ckpt')
    ckpt_dir = os.path.dirname(ckpt_path)
    print('ckpt_path', ckpt_path)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if len(args.pretrained) == 0:
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest:
            initial_step = int(latest.split('-')[-1].split('.')[0])
            load_path = latest

    else:
        print('loading', args.pretrained, args.pretrained_epoch)
        load_path = os.path.join(args.pretrained, '-', args.pretrained_epoch, '.ckpt')

    lr_decay_steps = {}
    for i, x in enumerate(args.lr_steps.split(',')):
        lr_decay_steps[int(x)] = args.lr*np.power(0.1, i+1)
    print('lr_steps', lr_decay_steps)
    lr_schedule = utils.get_lr_schedule(lr_decay_steps)
    init_lr = lr_schedule(initial_step, args.lr/strategy.num_replicas_in_sync)

    classifier.compile(optimizer=keras.optimizers.SGD(lr=init_lr, momentum=args.mom),
                       loss=MarginSoftmaxSparseCategoricalCrossentropy(config.num_classes,
                                                                       config.loss_s,
                                                                       config.loss_m1,
                                                                       config.loss_m2,
                                                                       config.loss_m3),
                       metrics=[keras.metrics.SparseCategoricalAccuracy()])
    classifier.summary()
    if load_path:
        classifier.load_weights(load_path)
        print('weights load from: {}'.format(load_path))

    _callbacks = [callbacks.LearningRateSchedulerOnBatch(lr_schedule, steps_per_epoch=batches_per_epoch, verbose=1),
                  keras.callbacks.TensorBoard(ckpt_dir, update_freq=args.frequent),
                  callbacks.FaceRecognitionValidation(extractor,
                                                      steps_per_epoch=batches_per_epoch,
                                                      valid_list=data_input.read_valid_sets(data_dir, config.val_targets),
                                                      period=args.verbose,
                                                      verbose=1),
                  callbacks.ModelCheckpointOnBatch(ckpt_path,
                                                   steps_per_epoch=batches_per_epoch,
                                                   monitor='score',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode='max',
                                                   period=args.verbose)]

    classifier.fit(train_dataset,
                   initial_epoch=initial_step // batches_per_epoch,
                   epochs=default.end_epoch,
                   steps_per_epoch=batches_per_epoch,
                   callbacks=_callbacks)


def main():
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    # tf_config = tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions())
    # tf_config.gpu_options.allow_growth = True
    # sess = tf.Session(config=tf_config)
    # K.set_session(sess)
    main()
