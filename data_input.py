import os
import pickle

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import tensorflow as tf

from scipy import misc

TRAIN_SET_NUM = 5822653


def train_parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int32)
    return img, label


def get_valid_parse_function(flip):
    def valid_parse_function(_bin):
        img = tf.image.decode_jpeg(_bin)
        img = tf.image.resize_images(img, [112, 112])
        img = tf.cast(img, dtype=tf.float32)
        if flip:
            img = tf.image.flip_left_right(img)
        return img
    return valid_parse_function


def training_dataset(tf_record_path, batch_size=128, shuffle_buffer=50000):
    dataset = tf.data.TFRecordDataset(tf_record_path)
    dataset = dataset.map(train_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle_buffer is not 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    batch_num = TRAIN_SET_NUM // batch_size
    return dataset, batch_num


def get_training_pipeline(tf_record_path, batch_size=128, shuffle_buffer=50000):
    dataset, _ = training_dataset(tf_record_path, batch_size, shuffle_buffer)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element


def count_training_data():
    sess = tf.Session()
    tf_records = os.path.join('data', 'train.tfrecords')
    if not os.path.exists(tf_records):
        raise FileExistsError(tf_records)
    batch_size = 1000
    iterator, next_element = get_training_pipeline(tf_records, batch_size, 0)
    sess.run(iterator.initializer)
    dataset_size = 0
    steps = 0
    while True:
        try:
            images, labels = sess.run(next_element)
            dataset_size += images.shape[0]
            if images.shape[0] % 1000 != 0:
                print('last batch size:', images.shape[0])
            steps += 1
            if steps % 100 == 0:
                print('steps', steps)
        except tf.errors.OutOfRangeError:
            print("Dataset size:", dataset_size)
            break


def view_training_data():
    sess = tf.Session()
    tf_records = os.path.join('data/faces_emore', 'train.tfrecords')
    if not os.path.exists(tf_records):
        raise FileExistsError(tf_records)
    iterator, next_element = get_training_pipeline(tf_records)
    sess.run(iterator.initializer)
    while True:
        try:
            images, labels = sess.run(next_element)
            one = images[0]
            misc.imsave('sample.jpg', one)
            print(one)
            images /= 255.
            plt.figure()
            for k in range(16):
                plt.subplot(4, 4, k+1)
                plt.imshow(images[k, ...])
                plt.text(0, 15, labels[k], fontdict={'color': 'red'})
                plt.title(labels[k])
            plt.show()
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break


def load_eval_data(dataset_list, image_size, path='data/faces_emore'):
    ver_list = []
    ver_name_list = []
    for db in dataset_list:
        print('begin db %s convert.' % db)
        data_set = load_bin(os.path.join(path, db+'.bin'), image_size)
        ver_list.append(data_set)
        ver_name_list.append(db)


def read_bin(path):
    try:
        with open(path, 'rb') as f:
            bins, is_same_list = pickle.load(f)  # py2
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            bins, is_same_list = pickle.load(f, encoding='bytes')
    if isinstance(bins, list):
        if len(bins) > 0:
            if isinstance(bins[0], np.ndarray):
                bins = [b.tobytes() for b in bins]
    return bins, is_same_list


def load_bin(path, image_size):
    bins, is_same_list = read_bin(path)
    data_list = []
    for _ in [0, 1]:
        data = np.empty((len(is_same_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(is_same_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)

        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img.asnumpy()
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, is_same_list


def view_bin(path):
    bins, is_same_list = read_bin(path)
    dataset = tf.data.Dataset.from_tensor_slices(bins)

    def parse(_bin):
        img = tf.image.decode_jpeg(_bin)
        img = tf.image.resize_images(img, [112, 112])
        img = tf.cast(img, dtype=tf.float32)
        img_flip = tf.image.flip_left_right(img)
        return img, img_flip

    dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(128)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()
    images, images_flip = sess.run(next_element)
    images /= 255.
    images_flip /= 255.
    images = np.concatenate((images, images_flip), 1)
    plt.figure()
    for k in range(16):
        plt.subplot(4, 4, k + 1)
        plt.imshow(images[k, ...])
        # plt.text(0, 15, labels[k], fontdict={'color': 'red'})
        # plt.title(labels[k])
    plt.show()


def read_valid_sets(data_dir, dataset_list):
    valid_set = {}
    for name in dataset_list:
        path = os.path.join(data_dir, name + ".bin")
        if os.path.exists(path):
            bins, is_same_list = read_bin(path)
            valid_set[name] = (bins, is_same_list)
            print('valid set', name)
    return valid_set


def make_valid_set(path, name, batch_size=1024):
    source = os.path.join(path, name + '.bin')
    if os.path.exists(source):
        bins, is_same_list = read_bin(source)
        dataset = tf.data.Dataset.from_tensor_slices(bins)\
            .map(get_valid_parse_function(False), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .batch(batch_size)
        dataset_flip = tf.data.Dataset.from_tensor_slices(bins) \
            .map(get_valid_parse_function(True), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(batch_size)

        return dataset, dataset_flip, is_same_list


def load_valid_set(data_dir, dataset_list):
    print('load valid set', dataset_list, 'from', data_dir)
    valid_set = {}
    for name in dataset_list:
        data_set, data_set_flip, is_same_list = make_valid_set(data_dir, name)
        valid_set[name] = (data_set, data_set_flip, is_same_list)
        print('valid set {} loaded in tf dataset.'.format(name))
    return valid_set


if __name__ == '__main__':
    view_training_data()
    # count_training_data()
    # load_valid_set('data', ['lfw', 'cfp_fp', 'agedb_30'])
    # read_valid_sets('data', [['lfw', 'cfp_fp', 'agedb_30', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']])
    # view_bin('data/faces_emore/vgg2_fp.bin')
    # load_eval_data(['lfw', 'cfp_fp', 'agedb_30'], (112, 112))

