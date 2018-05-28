# encoding: utf-8
import tensorflow as tf
from util import FLAGS
import word_embedding
import os
import sys
import time

class Model(object):
    def __init__(self, mode="train"):
        if mode == "train":
            self.mode = True
        else:
            self.mode = False
        self.epochs = FLAGS.num_epochs
        self.batch_size = FLAGS.batch_size
        self.vocab_size = FLAGS.vocab_size
        self.embedding_size = FLAGS.embedding_size
        self.w2v = word_embedding.Word2Vec(self.vocab_size, self.embedding_size)
        self.learning_rate = FLAGS.learning_rate
        self.keep_prob = FLAGS.keep_prob
        self.restore = FLAGS.restore
        self.max_query_word = FLAGS.query_len_threshold
        self.max_doc_word = FLAGS.title_len_threshold
        self.num_docs = 2
        self.filter_size = FLAGS.filter_size
        self.train_file_path = FLAGS.train_dir
        self.tf_record_dir = FLAGS.tf_record_dir
        #self.file_list = [self.train_file_path+f for f in os.listdir(self.train_file_path)]
        self.train_file = self.train_file_path
        self.train_test_file = FLAGS.train_set
        self.val_file = FLAGS.dev_set
        self.vocab_path = FLAGS.vocab_path
        self.vectors_path = FLAGS.vectors_path
        self.model_name = "{}_lr{}_filter{}_bz{}".format(FLAGS.flag, self.learning_rate, self.filter_size, self.batch_size)
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
        self.local_output = None
        self.distrib_output = None
        self.optimizer(self.features_local, self.queries, self.docs)
        #self.test(self.feature_local, self.query, self.doc)
        #self.merged_summary_op = tf.summary.merge([self.sm_loss_op, self.sm_emx_op])
        #self.merged_summary_op = tf.summary.merge([self.sm_loss_op])

    def _input_layer(self):
        #with tf.variable_scope('Inputs'):
        with tf.variable_scope('Train_Inputs'):
            self.features_local = tf.placeholder(dtype=tf.float32,
                                                 shape=(None, self.num_docs, self.max_query_word, self.max_doc_word),
                                                 name='features_local')
            self.queries = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='queries')
            self.docs = tf.placeholder(dtype=tf.int32, shape=(None, self.num_docs, self.max_doc_word), name='docs')
            #self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.num_docs], name='labels')

        with tf.variable_scope('Test_Inputs'):
            self.feature_local = tf.placeholder(dtype=tf.float32,
                                                shape=(None, self.max_query_word, self.max_doc_word),
                                                name='feature_local')
            self.query = tf.placeholder(dtype=tf.int32, shape=(None, self.max_query_word), name='query')
            self.doc = tf.placeholder(dtype=tf.int32, shape=(None, self.max_doc_word), name='doc')

    def _embed_layer(self, query, doc):
        with tf.variable_scope('Embedding_layer'), tf.device("/cpu:0"):
            '''
            self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                                    initializer=self.word_vec_initializer,
                                                    dtype=tf.float32,
                                                    trainable=False)
            '''
            self.embedding_matrix = self.w2v.id2embedding
            self.sm_emx_op = tf.summary.histogram('EmbeddingMatrix', self.embedding_matrix)
            embedding_query = tf.nn.embedding_lookup(self.embedding_matrix, query)
            embedding_doc = tf.nn.embedding_lookup(self.embedding_matrix, doc)
            return embedding_query, embedding_doc

    def local_model(self, feature_local, is_training=True, reuse=False):
        with tf.variable_scope('local_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            features_local = tf.reshape(feature_local, [-1, self.max_query_word, self.max_doc_word])
            conv = tf.layers.conv1d(inputs=features_local, filters=self.filter_size, kernel_size=[1],
                                    activation=tf.nn.tanh) #[?,max_query_word,1,self.filter_size]
            conv = tf.reshape(conv, [-1,self.filter_size*self.max_query_word]) #[?,max_query_word*self.filter_size]
            dense1 = tf.layers.dense(inputs=conv, units=self.filter_size, activation=tf.nn.tanh) #[?, self.filter_size]
            #dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob, training=is_training) #extra add
            dense2 = tf.layers.dense(inputs=dense1, units=self.filter_size, activation=tf.nn.tanh)
            self.local_output = dense2
            return self.local_output

    def distrib_model(self, query, title, is_training=True, reuse=False):
        with tf.variable_scope('Distrib_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            embedding_query, embedding_title = self._embed_layer(query=query, doc=title)
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
                title = tf.reshape(embedding_title, [-1, self.max_doc_word, self.embedding_size, 1])
                conv1 = tf.layers.conv2d(inputs=title, filters=self.filter_size,
                                         kernel_size=[3, self.embedding_size],
                                         activation=tf.nn.tanh, name="conv_title")  # [?,15-3+1,1, self.filter_size]
                pooling_size = self.max_doc_word - 3 + 1
                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[pooling_size, 1],
                                                strides=[1, 1], name="pooling_title")  # [?, 1,1 self.filter_size]
                pool1 = tf.reshape(pool1, [-1, self.filter_size])  # [?, self.filter_size]
                dense1 = tf.layers.dense(inputs=pool1, units=self.filter_size, activation=tf.nn.tanh, name="fc_title")
                self.distrib_title = dense1  # [?, self.filter_size]


            distrib = tf.multiply(self.distrib_query, self.distrib_title) #[?, self.filter_size]
            distrib = tf.reshape(distrib,[-1,self.filter_size]) #[?, self.dims2]
            fuly1 = tf.layers.dense(inputs=distrib, units=self.filter_size, activation=tf.nn.tanh)
            #drop = tf.layers.dropout(inputs=fuly1, rate=self.keep_prob, training=is_training)  # extra add
            fuly2 = tf.layers.dense(inputs=fuly1, units=self.filter_size, activation=tf.nn.tanh)
            self.distrib_output = fuly2
            print("distrib_output:",self.distrib_output)
            '''
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
                # dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob, training=is_training) #extra add
                dense2 = tf.layers.dense(inputs=dense1, units=self.filter_size, activation=tf.nn.tanh)
                self.match_output = dense2

            return self.distrib_output, self.match_output
            '''
            return self.distrib_output

    def ensemble_model(self, features_local, query, doc, is_training=True, reuse=False):
        with tf.variable_scope('emsemble_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            #self.model_output = tf.add(self.local_model(is_training=is_training, features_local = features_local,\
                                                        #reuse=reuse),self.distrib_model(is_training=is_training, \
                                                        #query=query,doc=doc,reuse=reuse))
            self.model_output = tf.concat([self.local_model(is_training=is_training, feature_local=features_local, \
                                                            reuse=reuse),self.distrib_model(is_training=is_training, \
                                                            query=query,title=doc,reuse=reuse)], axis=-1)
            fuly = tf.layers.dense(inputs=self.model_output, units=self.filter_size, activation=tf.nn.tanh)
            fuly1 = tf.layers.dense(inputs=fuly, units=1, activation=tf.nn.sigmoid)
            #self.model_output =  self.distrib_model(is_training=is_training, query=query, doc=doc,reuse=reuse)
            #self.model_output = self.local_model(is_training=is_training, features_local = features_local,reuse=reuse)

        #output = self.model_output
        output = fuly1
        print("ensemble_output: ", output)
        return output

    def optimizer(self, features_local, queries, docs):
        docs = tf.transpose(docs, [1, 0, 2], name="docs_transpose")  # [2, batch_size, words_num]
        features_local = tf.transpose(features_local, [1, 0, 2, 3], name="local_features_transpose")  # [2, batch_size, query_length, doc_length]
        print("docs: ", docs)
        print("feature_local: ", features_local)
        self.score_pos = self.ensemble_model(features_local=features_local[0], query=queries,
                                             doc=docs[0], is_training=True, reuse=False)  # [batch_size, 1]
        self.score_neg = self.ensemble_model(features_local=features_local[1], query=queries,
                                             doc=docs[1], is_training=True, reuse=True)  # [batch_size, 1]
        #with tf.variable_scope("Optimizer"):
        with tf.name_scope("loss"):
            self.score_pos = tf.squeeze(self.score_pos, -1, name="squeeze_pos")  # [batch_size]
            self.score_neg = tf.squeeze(self.score_neg, -1, name="squeeze_neg")  # [batch_size]
            self.sub = tf.subtract(self.score_pos, self.score_neg, name="pos_sub_neg")

            self.losses = tf.maximum(0.0, tf.subtract(0.5, tf.subtract(self.score_pos, self.score_neg)))
            self.loss = tf.reduce_mean(self.losses)
            #self.loss = tf.reduce_mean(tf.log(1.0 + tf.exp(- 2.0 * self.sub)))
            self.sm_loss_op = tf.summary.scalar('Loss', self.loss)

        with tf.name_scope("optimizer"):
            #self.optimize_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.optimize_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.90, beta2=0.999,epsilon=1e-08).minimize(self.loss)
        self.var_list = tf.trainable_variables()

    #def predict_score(self, feature_local, query, doc):

    def train(self, sess):
        print("training: ")
        init = tf.global_variables_initializer()
        sess.run(init)
        print("inited sess")
        self.w2v.load_word_file(self.vocab_path)
        self.w2v.load_tensorflow_embeddings(sess, self.vectors_path)
        self.saver = tf.train.Saver()
        if self.restore:
            self.saver.restore(sess=sess, save_path=self.save_path)
        acc_before = 0.0
        log_file = open(self.log_path + 'output.txt', 'w')
        for i in range(self.epochs):
            steps = 0
            batch_loss = 0
            for batch in self.w2v.get_train_batch(self.train_file, self.batch_size, self.max_query_word, self.max_doc_word):
                query, titles, qt_match = batch
                feed_dict = {self.queries: query,
                             self.docs: titles,
                             self.features_local: qt_match}
                #_, loss, summary_= sess.run([self.opt, self.loss, self.sm_loss_op], feed_dict=feed_dict)
                _, sub, pos_score, neg_score, loss= sess.run([self.optimize_op,self.sub, self.score_pos, self.score_neg, self.loss], feed_dict=feed_dict)
                batch_loss += loss
                #summary_writer.add_summary(summary=summary_, global_step=steps)
                self.saver.save(sess=sess, save_path=self.save_path + self.model_name)
                if steps % 200 == 0:
                    localtime = time.asctime(time.localtime(time.time()))
                    print("localtime: ", localtime)
                    print("epoch: ", i)
                    print("steps: ", steps)
                    print("on training set: ")
                    print("batch_loss:　", batch_loss/200)
                    print("sub: ", sub[:10])
                    print("pos_score: ", pos_score[:10])
                    print("neg_score: ", neg_score[:10])
                    self.saver.save(sess=sess, save_path=self.save_path + self.model_name)
                    self.mode = False
                    print("on validation set: ")
                    acc = self.eval(sess=sess, data_mode='valid')
                    print("accuracy: ", acc)
                    log_file.write("local_time: "+str(localtime)+ '\n')
                    log_file.write("on training set\n" + "batch_loss: "+ str(batch_loss/200 )+'\n')
                    log_file.write("pos_score: "+ str(pos_score[:10])+"\n")
                    log_file.write('neg_score: '+ str(neg_score[:10])+ '\n')
                    log_file.write("on valid set\n" + "acc: " + str(acc) + "\n"+"acc_before: " + str(acc_before) + "\n")
                    sys.stdout.flush()
                    if acc > acc_before:
                        acc_before = acc
                        self.saver.save(sess=sess, save_path=self.best_save_path + self.model_name)
                    self.mode = True
                    batch_loss = 0
                steps += 1

    def eval(self, sess, in_train = True, data_mode='valid'):
        print("eval:　")
        print("=====================")
        tf.global_variables_initializer().run()
        print("variables initialized")
        print(in_train, data_mode)
        if in_train == False:
            self.w2v.load_word_file(self.vocab_path)
            self.w2v.load_tensorflow_embeddings(sess, self.vectors_path)
            f = open("../data/predict_eval.txt", 'w')
        saver = tf.train.Saver(var_list=self.var_list)
        if data_mode == "test":
            saver.restore(sess=sess, save_path=self.best_save_path + self.model_name)
        else:
            saver.restore(sess=sess, save_path=self.save_path + self.model_name )
        print("data loaded: ")
        right = 0.0
        sum_count = 0.0
        flag = 0
        cnt = 0
        for batch in self.w2v.get_eval_batch(self.val_file, self.batch_size, self.max_query_word, self.max_doc_word):
            query,titles,qt_match= batch

            feed_dict = {self.queries: query,
                         self.docs: titles,
                         self.features_local: qt_match}
            sub, pos_score, neg_score, embedding_matrix = sess.run([self.sub, self.score_pos, self.score_neg, self.embedding_matrix], feed_dict=feed_dict)
            #print("sub: ", sub[0])
            '''
            if cnt == 0:
                with open("../data/word_embedding.txt", "w") as f:
                    for i in embedding_matrix:
                        f.write(" ".join(str(j) for j in i) + "\n")
                    cnt = 1
            '''
            for i, j, k in zip(sub, pos_score , neg_score):
                if flag == 0:
                    print("pos_score: ", pos_score[:10])
                    print("neg_score: ", neg_score[:10])
                    flag = 1
                sum_count += 1
                if i > 0:
                    right += 1
                try:
                    f.write(str(j) + '\t' + str(k) + '\n')
                except Exception as e:
                    pass
        acc = right / sum_count
        print("accuracy: ", acc)
        print("=========================")
        return acc


    def eval_qtitle(self, sess, data_mode="test"):
        print("eval qtitle10w:　")
        print("=====================")
        tf.global_variables_initializer().run()
        print("variables initialized")
        self.w2v.load_word_file(self.vocab_path)
        self.w2v.load_tensorflow_embeddings(sess, self.vectors_path)
        saver = tf.train.Saver(var_list=self.var_list)
        if data_mode == "test":
            saver.restore(sess=sess, save_path=self.best_save_path + self.model_name)
        else:
            saver.restore(sess=sess, save_path=self.save_path + self.model_name)
        print("data loaded: ")
        right = 0.0
        pre_bottom = 0
        recall_bottom = 0
        sum_count = 0.0
        with open("../data/predict_qtitle10w.txt", 'w') as f:
            for batch in self.w2v.get_qtitle10w_batch("../data/qtitle10w.test.txt", self.batch_size, self.max_query_word, self.max_doc_word):
                query,titles, qt_match, label, index = batch
                #print("query: " , len(query))
                #print("title: ", len(titles))
                #print("qt_match: ", len(qt_match))

                feed_dict = {self.queries: query,
                             self.docs: titles,
                             self.features_local: qt_match}
                sub, pos_score, neg_score = sess.run([self.sub, self.score_pos, self.score_neg], feed_dict=feed_dict)
                # print("sub: ", sub[0])
                for i, j, k in zip(pos_score, label, index):
                    f.write(k + '\t' + j + '\t' + str(i) + '\n')
                    sum_count += 1
                    if i >= 0.3 and int(j) >=2:
                        right += 1
                    if i>= 0.3:
                        pre_bottom += 1
                    if int(j) >=2:
                        recall_bottom += 1
            acc = right / sum_count
            precision = right/pre_bottom
            recall = right/recall_bottom
            print("=========================")
            print("accuracy: ", acc)
            print("precision: ", precision)
            print("recall: ", recall)
            print("=========================")
            return acc

