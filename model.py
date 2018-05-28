# encoding: utf-8
import tensorflow as tf
from util import FLAGS
import word_embedding
import time
import os
import numpy as np


class Model(object):
    def __init__(self, mode):
        if mode == "train":
            self.mode = True
        else:
            self.mode = False
        self.batch_size = FLAGS.batch_size
        self.vocab_size = FLAGS.vocab_size
        self.num_gpus = FLAGS.num_gpus
        self.embedding_size = FLAGS.embedding_size
        self.w2v = word_embedding.Word2Vec(self.vocab_size, self.embedding_size)
        self.learning_rate = FLAGS.learning_rate
        self.keep_prob = FLAGS.keep_prob
        self.max_query_word = FLAGS.query_len_threshold
        self.max_title_word = FLAGS.title_len_threshold
        self.filter_size = FLAGS.filter_size
        self.vocab_path = FLAGS.vocab_path
        self.vectors_path = FLAGS.vectors_path
        self.query = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='query')
        self.pos_title = tf.placeholder(dtype=tf.int32, shape=(None, self.max_title_word), name='pos_title')
        self.neg_title = tf.placeholder(dtype=tf.int32, shape=(None, self.max_title_word), name='neg_title')
        self.build()
        #self.optimizer(self.features_local, self.queries, self.docs)
        #self.test(self.feature_local, self.query, self.doc)
        #self.merged_summary_op = tf.summary.merge([self.sm_loss_op, self.sm_emx_op])
        #self.merged_summary_op = tf.summary.merge([self.sm_loss_op]
    '''
    def input_layer(self):
        with tf.variable_scope('Inputs'):
            self.query = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='query')
            self.pos_title = tf.placeholder(dtype=tf.int32, shape=(None, self.max_title_word), name='pos_title')
            self.neg_title = tf.placeholder(dtype=tf.int32, shape=(None, self.max_title_word), name='neg_title')
    '''
    def embedding_layer(self):
        with tf.variable_scope('Embedding'), tf.device("/cpu:0"):
            self.embedding_matrix = self.w2v.id2embedding
            self.sm_emx_op = tf.summary.histogram(name='EmbeddingMatrix', values=self.embedding_matrix)

    def feature_detection_layer(self, sentence, name=None, reuse=False, is_training=True):
        with tf.variable_scope('Feature_detection {}:'.format(name)):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            embedding_sent = tf.nn.embedding_lookup(self.embedding_matrix, sentence)
            sent = tf.reshape(embedding_sent, [-1, self.max_query_word, self.embedding_size, 1])  # [?, max_len, self.embedding_size,1]
            conv1 = tf.layers.conv2d(inputs=sent, filters=self.filter_size,
                                     kernel_size=[3, self.embedding_size],
                                     activation=tf.nn.tanh, name="conv_{}".format(name))  # [?,max_len-3+1,1, self.filter_size]
            pooling_size = self.max_query_word - 3 + 1
            pooling = tf.layers.max_pooling2d(inputs=conv1,
                                           pool_size=[pooling_size, 1],
                                           strides=[1, 1],
                                           name="pooling_{}".format(name))  # [?, 1,1 self.filter_size]
            pooling = tf.reshape(pooling, [-1, self.filter_size])  # [?, self.filter_size]
            dense = tf.layers.dense(inputs=pooling, units=self.filter_size, activation=tf.nn.tanh, name="fc_name".format(name))
            return dense  # [?, self.filter_size]

    def match_layer(self, query_feature, title_feature, reuse=False, is_training=True):
        match_vector = tf.multiply(query_feature, title_feature)  # [?, self.filter_size]
        match_vector = tf.reshape(match_vector, [-1, self.filter_size])  # [?, self.dims2]
        fc1 = tf.layers.dense(inputs=match_vector, units=self.filter_size, activation=tf.nn.tanh)
        drop = tf.layers.dropout(inputs=fc1, rate=self.keep_prob, training=is_training)  # extra add
        fc2 = tf.layers.dense(inputs=drop, units=self.filter_size, activation=tf.nn.tanh)
        match_vector = fc2
        print("match_vector:", match_vector)

        with tf.variable_scope("Match"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            title_feature_reshape = tf.transpose(title_feature, perm=[0, 2, 1])
            match_matrix = tf.matmul(query_feature, title_feature_reshape)
            print("query_feature", query_feature)
            print("title_feature", title_feature)
            print("match_matrix", match_matrix)
            conv = tf.layers.conv1d(inputs=match_matrix,
                                    filters=self.filter_size,
                                    kernel_size=[1],
                                    activation=tf.nn.tanh)  # [?,max_len,1,self.filter_size]
            conv = tf.reshape(conv, [-1, self.filter_size * self.max_query_word])  # [?,max_len*self.filter_size]
            dense1 = tf.layers.dense(inputs=conv,
                                     units=self.filter_size,
                                     activation=tf.nn.tanh)  # [?, self.filter_size]
            dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob, training=is_training)  # extra add
            dense2 = tf.layers.dense(inputs=dropout, units=self.filter_size, activation=tf.nn.tanh)
            match_output = dense2

        return tf.concat([match_vector, match_output], axis=-1)

    def fully_connected_layer(self, feature, reuse=False, is_training=True):
        with tf.variable_scope("Fc"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            fc_1 = tf.layers.dense(inputs=feature,
                                  units=self.filter_size,
                                  activation=tf.nn.tanh)  # [?, self.filter_size]
            fc_2 = tf.layers.dense(inputs=fc_1,
                                   units=1,
                                   activation=tf.nn.tanh)  # [?, self.filter_size]
            return fc_2

    def forword(self, is_training=True):
        query_feature = self.feature_detection_layer(self.query, name="query", reuse=False, is_training=is_training)
        pos_title_feature = self.feature_detection_layer(self.pos_title, name="title", reuse=False, is_training=is_training)
        neg_title_feature = self.feature_detection_layer(self.neg_title, name="title", reuse=True, is_training=is_training)
        pos_match_feature = self.match_layer(query_feature, pos_title_feature, reuse=False, is_training=is_training)
        neg_match_feature = self.match_layer(query_feature, neg_title_feature, reuse=True, is_training=is_training)

        score_pos = tf.squeeze(self.fully_connected_layer(pos_match_feature, reuse=False, is_training=is_training),
                                    axis=-1,
                                    name="squeeze_pos")  # [batch_size, ]
        score_neg = tf.squeeze(self.fully_connected_layer(neg_match_feature, reuse=True, is_training=is_training),
                                    axis=-1,
                                    name="squeeze_neg")  # [batch_size, ]
        sub = tf.subtract(self.score_pos, self.score_neg, name="sub")

        with tf.name_scope("loss"):
            losses = tf.maximum(0.0, tf.subtract(1.0, sub))
            print("build loss: ")
            loss = tf.reduce_mean(losses)
            #self.loss = tf.reduce_mean(tf.log(1.0 + tf.exp(- 2.0 * self.sub)))
            self.sm_loss_op = tf.summary.scalar('Loss', self.loss)

        with tf.name_scope("optimizer"):
            #self.optimize_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.90, beta2=0.999,epsilon=1e-08)
            #grad_and_vars = self.optimizer.compute_gradients(loss=self.loss)
            #for grad, var in grad_and_vars:
            #    tf.summary.histogram(name='grad_' + var.name, values=grad)
            #self.opt = self.optimizer.apply_gradients(grads_and_vars=grad_and_vars)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.90, beta2=0.999,epsilon=1e-08).minimize(self.loss)
            summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(logdir=self.log_path)

    def build_model_multi_gpu(self):
        tower_grads = []
        #reuse_vars = False
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        for i in range(self.num_gpus):
            with tf.device('/gpu:{}' .format(i) ):
                query = self.query[i * self.batch_size: (i + 1) * self.batch_size]
                pos_title = self.pos_title[i * self.batch_size: (i + 1) * self.batch_size]
                neg_title = self.neg_title[i * self.batch_size: (i + 1) * self.batch_size]
                query_embedding = tf.nn.embedding_lookup(self.embedding_matrix, query)
                pos_title_embedding = tf.nn.embedding_lookup(self.embedding_matrix, pos_title)
                neg_title_embedding = tf.nn.embedding_lookup(self.embedding_matrix, neg_title)
            _labels_ph = self.labels[i * self.config.batch_size: (i + 1) * self.config.batch_size]
            model_out = self.build_model(reuse_vars, question_embedding, answer_embedding, _mask_q_ph, _mask_a_ph)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_out, labels=_labels_ph))
        grads = optimizer.compute_gradients(self.loss)
        if i == 0:
            test_out = self.build_model(True, question_embedding, answer_embedding, _mask_q_ph, _mask_a_ph)
            self.predict_probability = tf.nn.softmax(test_out)
            self.predict_val = tf.argmax(test_out, 1)
            correct_pred = tf.equal(self.predict_val, tf.argmax(_labels_ph, 1))
            self.accuracy_val = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        reuse_vars = True
        tower_grads.append(grads)


tower_grads = self.average_gradients(tower_grads)
self.train_op = optimizer.apply_gradients(tower_grads)