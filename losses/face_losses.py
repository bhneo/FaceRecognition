import tensorflow as tf
import time
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.losses import Loss
from tensorflow.python.ops.losses import losses_impl
from tensorflow import keras
from tensorflow.python.keras.layers import Layer
from config import config as cfg


def make_logits(embedding, label_one_hot, class_num, loss_type='margin_softmax', s=64.0, m1=1.0, m2=0.5, m3=0.0, w=None, use_bias=False):
    embedding_size = embedding.get_shape().as_list()[-1]
    if w is None:
        w = tf.Variable(tf.random_normal([embedding_size, class_num], stddev=0.01), name='fc7_weight')
    if loss_type == 'margin_softmax':
        embedding_norm = tf.norm(embedding, axis=-1, keepdims=True, name='fc1n')
        embedding = embedding/embedding_norm
        w_norm = tf.norm(w, axis=0, keepdims=True)
        w = w/w_norm
        embedding_norm_scale = embedding * s
        fc7 = tf.matmul(embedding_norm_scale, w, name='fc7')
        if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
            if m1 == 1.0 and m2 == 0.0:
                s_m = s * m3
                label_one_hot = label_one_hot * s_m
                fc7 = fc7 - label_one_hot
            else:
                cos_t = fc7 / s
                t = tf.math.acos(cos_t)
                if m1 != 1.0:
                    t = t * m1
                if m2 > 0.0:
                    t = t + m2
                body = tf.math.cos(t)
                if m3 > 0.0:
                    body = body - m3
                diff = body * s - fc7
                body = tf.multiply(label_one_hot, diff)
                fc7 = fc7 + body
    else:
        fc7 = tf.matmul(embedding, w)
        if use_bias:
            bias = tf.Variable(tf.zeros([class_num, ]), name='fc7_bias')
            fc7 = tf.add(fc7, bias)
    return fc7


def make_logits_v2(embedding, one_hot_label, class_num, loss_type='margin_softmax', s=64.0, m1=1.0, m2=0.5, m3=0.0, w=None, use_bias=False):
    embedding_size = embedding.get_shape().as_list()[-1]
    if w is None:
        w = tf.Variable(tf.random_normal([embedding_size, class_num], stddev=0.01), name='fc7_weight')
    if loss_type == 'margin_softmax':
        embedding_norm = tf.norm(embedding, axis=-1, keepdims=True, name='fc1n')
        embedding = embedding / embedding_norm
        w_norm = tf.norm(w, axis=0, keepdims=True)
        w = w / w_norm
        embedding_norm_scale = embedding * s
        fc7 = tf.matmul(embedding_norm_scale, w, name='fc7')
        if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
            if m1 == 1.0 and m2 == 0.0:
                s_m = s * m3
                label_one_hot = one_hot_label * s_m
                fc7 = fc7 - label_one_hot
            else:
                cos_t = fc7 / s
                t = tf.math.acos(cos_t)
                if m1 != 1.0:
                    t = t * m1
                if m2 > 0.0:
                    t = t + m2
                body = tf.math.cos(t)
                if m3 > 0.0:
                    body = body - m3
                body = body * s
                mask = 1 - one_hot_label
                fc7 = fc7*mask + body*one_hot_label
    else:
        fc7 = tf.matmul(embedding, w)
        if use_bias:
            bias = tf.Variable(tf.zeros([class_num, ]), name='fc7_bias')
            fc7 = tf.add(fc7, bias)
    return fc7


def make_logits_v3(embedding, label_one_hot, class_num, loss_type='margin_softmax', s=64.0, m1=1.0, m2=0.5, m3=0.0, w=None, use_bias=False):
    embedding_size = embedding.get_shape().as_list()[-1]
    if w is None:
        w = tf.Variable(tf.random_normal([embedding_size, class_num], stddev=0.01), name='fc7_weight')
    if loss_type == 'margin_softmax':
        embedding_norm = tf.norm(embedding, axis=-1, keepdims=True, name='fc1n')
        embedding = embedding/embedding_norm
        w_norm = tf.norm(w, axis=0, keepdims=True)
        w = w/w_norm
        embedding_norm_scale = embedding * s
        fc7 = tf.matmul(embedding_norm_scale, w, name='fc7')
        if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
            if m1 == 1.0 and m2 == 0.0:
                s_m = s * m3
                label_one_hot = label_one_hot * s_m
                fc7 = fc7 - label_one_hot
            else:
                cos_t = tf.reduce_sum(fc7 * label_one_hot, -1) / s
                t = tf.math.acos(cos_t)
                if m1 != 1.0:
                    t = t * m1
                if m2 > 0.0:
                    t = t + m2
                body = tf.math.cos(t)
                if m3 > 0.0:
                    body = body - m3
                diff = tf.expand_dims(body * s, -1) * label_one_hot - fc7
                body = tf.multiply(label_one_hot, diff)
                fc7 = fc7 + body
    else:
        fc7 = tf.matmul(embedding, w)
        if use_bias:
            bias = tf.Variable(tf.zeros([class_num, ]), name='fc7_bias')
            fc7 = tf.add(fc7, bias)
    return fc7


def cal_norm(inputs, w):
    embedding_norm = tf.norm(inputs, axis=-1, keepdims=True, name='fc1n')
    embedding = inputs / embedding_norm
    w_norm = tf.norm(w, axis=0, keepdims=True)
    w = w / w_norm
    out = tf.matmul(embedding, w, name='out')
    return out


class MarginSoftmaxSparseCategoricalCrossentropy(Loss):
    def __init__(self,
                 units,
                 s=64.0,
                 m1=1.0,
                 m2=0.5,
                 m3=0.0,
                 reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE,
                 name=None):
        super(MarginSoftmaxSparseCategoricalCrossentropy, self).__init__(
            reduction=reduction, name=name)
        self.units = units
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def call(self, y_true, y_pred):
        y_one_hot = tf.one_hot(tf.squeeze(tf.to_int32(y_true)), self.units)

        if self.m1 != 1.0 or self.m2 != 0.0 or self.m3 != 0.0:
            if self.m1 == 1.0 and self.m2 == 0.0:
                y_pred = self.s * (y_pred - y_one_hot * self.m3)
            else:
                cos_t = tf.reduce_sum(y_pred * y_one_hot, -1)
                t = tf.math.acos(cos_t)
                if self.m1 != 1.0:
                    t = t * self.m1
                if self.m2 > 0.0:
                    t = t + self.m2
                body = tf.math.cos(t)
                if self.m3 > 0.0:
                    body = body - self.m3
                diff = tf.expand_dims(body, -1) * y_one_hot - y_pred
                body = tf.multiply(y_one_hot, diff)
                y_pred = self.s * (y_pred + body)

        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=y_pred)


if __name__ == '__main__':
    cfg.debug = True
    tf.enable_eager_execution()
    batch_num = 100
    batch_size = 2048
    feature_dim = 512
    persons = 100
    embeddings = tf.constant(tf.random_normal([batch_size, feature_dim]))
    ws = tf.Variable(tf.random_normal([feature_dim, persons]))
    labels = tf.constant(tf.random_uniform([batch_size, ], maxval=persons, dtype=tf.int32))
    labels_one_hot = tf.one_hot(labels, persons)

    tmp = make_logits(embeddings, labels_one_hot, persons, w=ws)

    start = time.time()
    for _ in range(batch_num):
        labels_one_hot = tf.one_hot(labels, persons)
        out1 = make_logits(embeddings, labels_one_hot, persons, w=ws)
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out1, labels=labels))
    t1 = time.time()
    print('logits 1 cost:', t1-start, 's')
    print('loss1:', loss1)
    for _ in range(batch_num):
        labels_one_hot = tf.one_hot(labels, persons)
        out2 = make_logits_v2(embeddings, labels_one_hot, persons, w=ws)
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out2, labels=labels))
    t2 = time.time()
    print('logits 2 cost:', t2 - t1, 's')
    print('loss2:', loss2)
    for _ in range(batch_num):
        labels_one_hot = tf.one_hot(labels, persons)
        out3 = make_logits_v3(embeddings, labels_one_hot, persons, w=ws)
        loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out3, labels=labels))
    t3 = time.time()
    print('logits 3 cost:', t3 - t2, 's')
    print('loss3:', loss3)
    _loss = MarginSoftmaxSparseCategoricalCrossentropy(persons)
    t3 = time.time()
    for _ in range(batch_num):
        out4 = cal_norm(embeddings,  ws)
        loss4 = _loss(labels, out4)
    t4 = time.time()
    print('logits 4 cost:', t4 - t3, 's')
    print('loss4:', loss4)


