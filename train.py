import argparse
import os

import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

import data_input
import verification
from nets import fmobilefacenet
from common import block, utils
from config import config, default, generate_config


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

    label = keras.Input(shape=(1,), name='label', dtype=tf.int32)
    fc7 = block.FaceCategoryOutput(config.num_classes, loss_type=config.loss_name, s=config.loss_s, m1=config.loss_m1, m2=config.loss_m2, m3=config.loss_m3)((embedding, label))
    classifier = keras.Model(inputs=(data, label), outputs=fc7, name='classifier')

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
    train_dataset, batches_per_epoch = data_input.training_dataset(training_path, default.per_batch_size)

    extractor, classifier = build_model((image_size[0], image_size[1], 3), args)

    global_step = 0
    ckpt_path = os.path.join(args.models_root, '%s-%s-%s' % (args.network, args.loss, args.dataset), 'model-{step:04d}.ckpt')
    ckpt_dir = os.path.dirname(ckpt_path)
    print('ckpt_path', ckpt_path)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if len(args.pretrained) == 0:
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest:
            global_step = int(latest.split('-')[-1].split('.')[0])
            classifier.load_weights(latest)
    else:
        print('loading', args.pretrained, args.pretrained_epoch)
        load_path = os.path.join(args.pretrained, '-', args.pretrained_epoch, '.ckpt')
        classifier.load_weights(load_path)

    initial_epoch = global_step // batches_per_epoch
    rest_batches = global_step % batches_per_epoch

    lr_decay_steps = [(int(x), args.lr*np.power(0.1, i+1)) for i, x in enumerate(args.lr_steps.split(','))]
    print('lr_steps', lr_decay_steps)

    valid_datasets = data_input.load_valid_set(data_dir, config.val_targets)

    classifier.compile(optimizer=keras.optimizers.SGD(lr=args.lr, momentum=args.mom),
                       loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                       metrics=[keras.metrics.SparseCategoricalAccuracy()])
    classifier.summary()

    tensor_board = keras.callbacks.TensorBoard(ckpt_dir)
    tensor_board.set_model(classifier)

    train_names = ['train_loss', 'train_acc']
    train_results = []
    highest_score = 0
    for epoch in range(initial_epoch, default.end_epoch):
        for batch in range(rest_batches, batches_per_epoch+1):
            utils.update_learning_rate(classifier, lr_decay_steps, global_step)
            train_results = classifier.train_on_batch(train_dataset, reset_metrics=False)
            global_step += 1
            if global_step % 1000 == 0:
                print('lr-batch-epoch:', float(K.get_value(classifier.optimizer.lr)), batch, epoch)
            if global_step >= 0 and global_step % args.verbose == 0:
                acc_list = []
                for key in valid_datasets:
                    data_set, data_set_flip, is_same_list = valid_datasets[key]
                    embeddings = extractor.predict(data_set)
                    embeddings_flip = extractor.predict(data_set_flip)
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

                if len(acc_list) > 0:
                    score = sum(acc_list)
                    if highest_score == 0:
                        highest_score = score
                    elif highest_score >= score:
                        print('\nStep %05d: score did not improve from %0.5f' %
                              (global_step, highest_score))
                    else:
                        path = ckpt_path.format(step=global_step)
                        print('\nStep %05d: score improved from %0.5f to %0.5f,'
                              ' saving model to %s' % (global_step, highest_score,
                                                       score, path))
                        highest_score = score
                        classifier.save_weights(path)

        utils.write_log(tensor_board, train_names, train_results, epoch)
        classifier.reset_metrics()


def main():
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
