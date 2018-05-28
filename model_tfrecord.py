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
        self.epochs = FLAGS.num_epochs
        self.steps = 10000000000
        self.batch_size = FLAGS.batch_size
        self.vocab_size = FLAGS.vocab_size
        self.embedding_size = FLAGS.embedding_size
        self.w2v = word_embedding.Word2Vec(self.vocab_size, self.embedding_size)
        self.learning_rate = FLAGS.learning_rate
        self.keep_prob = FLAGS.keep_prob
        self.restore = FLAGS.restore
        self.max_query_word = FLAGS.query_len_threshold
        self.max_title_word = FLAGS.title_len_threshold
        self.num_titles = 2
        self.filter_size = FLAGS.filter_size
        self.train_file_path = FLAGS.train_dir
        self.tf_record_path = FLAGS.tf_record_dir
        self.file_list = [f for f in os.listdir(self.tf_record_path)]
        self.train_test_file = FLAGS.train_set
        self.val_file = FLAGS.dev_set
        self.vocab_path = FLAGS.vocab_path
        self.vectors_path = FLAGS.vectors_path
        self.model_name = "lr{}_filter{}_bz{}".format(self.learning_rate, self.filter_size, self.batch_size)
        #self.local_output = None
        #self.distrib_output = None
        self.log_path = FLAGS.log_path + self.model_name + '/'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.save_path = FLAGS.save_path + self.model_name + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.best_save_path = FLAGS.save_path + self.model_name + '-best' + '/'
        if not os.path.exists(self.best_save_path):
            os.makedirs(self.best_save_path)
        self._input_layer()
        self.build()
        #self.optimizer(self.features_local, self.queries, self.docs)
        #self.test(self.feature_local, self.query, self.doc)
        #self.merged_summary_op = tf.summary.merge([self.sm_loss_op, self.sm_emx_op])
        #self.merged_summary_op = tf.summary.merge([self.sm_loss_op])

    def _input_layer(self):
        with tf.variable_scope('Inputs'):
            self.features_local = tf.placeholder(dtype=tf.float32, shape=(None, self.num_titles, self.max_query_word, self.max_title_word), name='features_local')
            self.queries = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='queries')
            self.titles = tf.placeholder(dtype=tf.int32, shape=( self.num_titles, None,self.max_title_word), name='titles')

    def _embed_layer(self, query, title):
        with tf.variable_scope('Embedding_layer'), tf.device("/cpu:0"):
            self.embedding_matrix = self.w2v.id2embedding
            self.sm_emx_op = tf.summary.histogram(name='EmbeddingMatrix', values=self.embedding_matrix)
            embedding_query = tf.nn.embedding_lookup(self.embedding_matrix, query)
            embedding_title = tf.nn.embedding_lookup(self.embedding_matrix, title)
            return embedding_query, embedding_title

    def build_feature_local(self, query, pos_title, neg_title):
        qa_matchs = []
        qa_match = np.zeros((2, self.max_query_word, self.max_title_word))
        cnt = 0
        for q, t1, t2 in zip(query, pos_title, neg_title):
            #qa_match = [[0 for i in range(self.max_title_word)] for j in range(self.max_query_word)]
            for i, word_q in enumerate(q):
                for j, word_t in enumerate(t1):
                    if word_q == word_t and int(word_t) != 4200000 and int(word_t) != 0:
                        qa_match[0,i,j] = 1
            # qa_match = [[0 for i in range(self.max_title_word)] for j in range(self.max_query_word)]
            for i, word_q in enumerate(q):
                for j, word_t in enumerate(t2):
                    if word_q == word_t and int(word_t) != 4200000 and int(word_t) != 0:
                        qa_match[1,i,j] = 1
            qa_matchs.append(qa_match)
            cnt += 1
        #print("build_qt_match: ", np.shape(qa_matchs))
        return qa_matchs


    def local_model(self, feature_local, is_training=True, reuse=False):
        with tf.variable_scope('local_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            features_local = tf.reshape(feature_local, [-1, self.max_query_word, self.max_title_word])
            conv = tf.layers.conv1d(inputs=features_local, filters=self.filter_size, kernel_size=[1],
                                    activation=tf.nn.tanh) #[?,max_query_word,1,self.filter_size]
            conv = tf.reshape(conv, [-1,self.filter_size*self.max_query_word]) #[?,max_query_word*self.filter_size]
            dense1 = tf.layers.dense(inputs=conv, units=self.filter_size, activation=tf.nn.tanh) #[?, self.filter_size]
            dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob, training=is_training) #extra add
            dense2 = tf.layers.dense(inputs=dropout, units=self.filter_size, activation=tf.nn.tanh)
            self.local_output = dense2
            return self.local_output

    def distrib_model(self, query, title, is_training=True, reuse=False):
        with tf.variable_scope('Distrib_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            embedding_query, embedding_title = self._embed_layer(query=query, title=title)
            with tf.variable_scope('distrib_query'):
                query = tf.reshape(embedding_query,
                                   [-1, self.max_query_word, self.embedding_size, 1])  # [?, max_query_word(=15), self.embedding_size,1]
                conv1 = tf.layers.conv2d(inputs=query, filters=self.filter_size,
                                         kernel_size=[3, self.embedding_size],
                                         activation=tf.nn.tanh, name="conv_query")  # [?,15-3+1,1, self.filter_size]
                pooling_size = self.max_query_word - 3 + 1
                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[pooling_size, 1],
                                                strides=[1, 1], name="pooling_query")  # [?, 1,1 self.filter_size]
                pool1 = tf.reshape(pool1, [-1, self.filter_size])  # [?, self.filter_size]
                dense1 = tf.layers.dense(inputs=pool1, units=self.filter_size, activation=tf.nn.tanh, name="fc_query")
                self.distrib_query = dense1  # [?, self.filter_size]

            with tf.variable_scope('distrib_title'):
                title = tf.reshape(embedding_title, [-1, self.max_title_word, self.embedding_size, 1])
                conv1 = tf.layers.conv2d(inputs=title, filters=self.filter_size,
                                         kernel_size=[3, self.embedding_size],
                                         activation=tf.nn.tanh, name="conv_title")  # [?,15-3+1,1, self.filter_size]
                pooling_size = self.max_title_word - 3 + 1
                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[pooling_size, 1],
                                                strides=[1, 1], name="pooling_title")  # [?, 1,1 self.filter_size]
                pool1 = tf.reshape(pool1, [-1, self.filter_size])  # [?, self.filter_size]
                dense1 = tf.layers.dense(inputs=pool1, units=self.filter_size, activation=tf.nn.tanh, name="fc_title")
                self.distrib_title = dense1  # [?, self.filter_size]


            distrib = tf.multiply(self.distrib_query, self.distrib_title) #[?, self.filter_size]
            distrib = tf.reshape(distrib,[-1,self.filter_size]) #[?, self.dims2]
            fuly1 = tf.layers.dense(inputs=distrib, units=self.filter_size, activation=tf.nn.tanh)
            drop = tf.layers.dropout(inputs=fuly1, rate=self.keep_prob, training=is_training)  # extra add
            fuly2 = tf.layers.dense(inputs=drop, units=self.filter_size, activation=tf.nn.tanh)
            self.distrib_output = fuly2
            print("distrib_output:",self.distrib_output)

            with tf.variable_scope("distrib_match"):
                embedding_title = tf.transpose(embedding_title, perm=[0, 2, 1])
                match_matrix = tf.matmul(embedding_query, embedding_title)
                print("embedding query", embedding_query)
                print("embedding title",embedding_title)
                print("match_matrix", match_matrix)
                conv = tf.layers.conv1d(inputs=match_matrix, filters=self.filter_size, kernel_size=[1],
                                        activation=tf.nn.tanh)  # [?,max_query_word,1,self.filter_size]
                conv = tf.reshape(conv,
                                  [-1, self.filter_size * self.max_query_word])  # [?,max_query_word*self.filter_size]
                dense1 = tf.layers.dense(inputs=conv, units=self.filter_size,
                                         activation=tf.nn.tanh)  # [?, self.filter_size]
                dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob, training=is_training) #extra add
                dense2 = tf.layers.dense(inputs=dropout, units=self.filter_size, activation=tf.nn.tanh)
                self.match_output = dense2

            return self.distrib_output, self.match_output

    def ensemble_model(self, feature_local, query, title, is_training=True, reuse=False):
        with tf.variable_scope('emsemble_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            local_representation = self.local_model(is_training=is_training, feature_local=feature_local, reuse=reuse)
            distrib_representation, match_representation = self.distrib_model(is_training=is_training, query=query,title=title,reuse=reuse)
            self.model_output = tf.concat([local_representation, distrib_representation, match_representation], axis=-1)
            fuly = tf.layers.dense(inputs=self.model_output, units=self.filter_size, activation=tf.nn.tanh)
            fuly1 = tf.layers.dense(inputs=fuly, units=1, activation=tf.nn.sigmoid)
        output = fuly1
        return output

    def build(self):
        features_local = tf.transpose(self.features_local, [1, 0, 2, 3])
        #self.titles = tf.transpose(self.titles, [1, 0, 2])
        self.score_pos = tf.squeeze(self.ensemble_model(feature_local=features_local[0], query=self.queries,
                                                        title=self.titles[0], is_training=self.mode, reuse=False), -1,
                                    name="squeeze_pos")  # [batch_size, ]
        self.score_neg = tf.squeeze(self.ensemble_model(feature_local=features_local[1], query=self.queries,
                                                        title=self.titles[1], is_training=self.mode, reuse=True), -1,
                                    name="squeeze_neg")  # [batch_size, ]
        self.sub = tf.subtract(self.score_pos, self.score_neg, name="pos_sub_neg")

        self.var_list = tf.trainable_variables()
        for var in self.var_list:
            #print(var.name)
            tf.summary.histogram(name=var.name, values=var)

    def train(self, sess):
        print("training: ")
        with tf.name_scope("loss"):
            self.losses = tf.maximum(0.0, tf.subtract(1.0, self.sub))
            #self.loss = tf.reduce_mean(self.losses)
            print("build loss: ")
            self.loss = tf.reduce_mean(self.losses)
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

        query, pos_title, neg_title = self.w2v.load_tfrecord([self.tf_record_path + i for i in self.file_list])

        init = tf.global_variables_initializer()
        sess.run(init)
        print("inited sess")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        self.w2v.load_word_file(self.vocab_path)
        self.w2v.load_tensorflow_embeddings(sess, self.vectors_path)
        self.saver = tf.train.Saver()
        if self.restore:
            self.saver.restore(sess=sess, save_path=self.save_path)
        acc_before = 0.0
        for i in range(self.steps):
            start_time = time.time()
            steps = i
            batch_loss = 0.0
            query_, pos_title_, neg_title_ = sess.run([query, pos_title, neg_title])
            titles = np.array([pos_title_, neg_title_])
            qt_match = self.build_feature_local(query_, pos_title_, neg_title_)
            qt_match = np.reshape(np.array(qt_match), (-1, 2, self.max_query_word, self.max_title_word))
            #print("qt_match_shape: ", np.shape(qt_match))
            #print("query: ", query_)
            #print("titles")
            feed_dict = {self.queries: query_,
                         self.titles: titles,
                         self.features_local: qt_match}
            _, loss, summary_= sess.run([self.opt, self.loss, summary], feed_dict=feed_dict)
            batch_loss += loss
            summary_writer.add_summary(summary=summary_, global_step=steps)
            #self.saver.save(sess=sess, save_path=self.save_path + self.model_name)
            if steps % 300 == 0:
                print("steps: ", steps)
                print("on training set: ")
                print("batch_loss:　", batch_loss/300)
                self.saver.save(sess=sess, save_path=self.save_path + self.model_name)
                self.mode = False
                print("on validation set: ")
                acc = self.eval(sess=sess, data_mode='valid')
                print("accuracy: ", acc)
                if acc > acc_before:
                    acc_before = acc
                    self.saver.save(sess=sess, save_path=self.best_save_path + self.model_name)
                self.mode = True
                batch_loss = 0

        coord.request_stop()
        coord.join(threads)

    def eval(self, sess, data_mode='valid'):
        print("eval:　")
        print("=====================")
        tf.global_variables_initializer().run()
        print("variables initialized")
        saver = tf.train.Saver(var_list=self.var_list)
        if data_mode == "test":
            saver.restore(sess=sess, save_path=self.best_save_path + self.model_name)
        else:
            saver.restore(sess=sess, save_path=self.save_path + self.model_name )
        print("data loaded: ")
        right = 0.0
        sum_count = 0.0
        for batch in self.w2v.get_eval_batch(self.val_file, self.batch_size, self.max_query_word, self.max_title_word):
            query, pos_title, neg_title = batch
            titles = [pos_title, neg_title]
            qt_match = self.build_feature_local(query, pos_title, neg_title)
            qt_match = np.reshape(np.array(qt_match), (-1, 2, self.max_query_word, self.max_title_word))
            feed_dict = {self.queries: query,
                         self.titles: titles,
                         self.features_local: qt_match}
            sub = sess.run([self.sub], feed_dict=feed_dict)
            for i in sub[0]:
                sum_count += 1
                if i > 0:
                    right += 1
        acc = right/sum_count
        print("accuracy: ", acc)
        print("=========================")
        return acc