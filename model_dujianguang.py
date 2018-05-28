#coding=utf-8
import tensorflow as tf
import numpy as np
import word_embedding_shunshun as word_embedding
import os
import re
import string
import sys
import time
from tensorflow.python.framework import ops
class CONFIG():
    def __init__(self):
        self.vocab_size = 4200000
        self.embedding_size = 100
        self.vocab_size = 2765067
        self.embedding_size = 200
        self.max_q_length = 20
        self.max_a_length = 200
        self.learning_rate = 0.005
        self.hidden_size = 300
        self.dropout_keep_prob = 0.7
        self.classes = 2
        self.batch_size = 32
	    self.gpus = [0,1,2,3]
	    self.num_gpus = len(self.gpus)
        self.epoch = 10
        self.forget_bias = 1.0
        self.word_table_file = '/search/dujianguang/word_table.tensorflow'
        self.word_embedding_file = '/search/dujianguang/word_embedding.tensorflow'
        self.word_table_file = '/search/dujianguang/word_embedding/word_list.no_dict.baike'
        self.word_embedding_file = '/search/dujianguang/word_embedding/word_embedding.no_dict.baike'
        self.train_file = '../data/click3/train.data.1m'
        self.eval_file = '../data/click3/eval.data.100k'
        #self.test_file = '/search/dujianguang/lihaitao/qa_classify/lstm_code/data/wenwen/first_place_wenda/dumpres_wenda_result.qapair'
        #self.test_file = '/search/dujianguang/lihaitao/qa_classify/lstm_code/data/wenwen/first_place_wenda/20180103/qa_seg.file'
        #self.test_file = self.eval_file
        self.model_save_file = 'model_save_path/model_click3'
        #self.model_save_file = 'model_save_path/mix/model_18'
        #self.test_out_file = '/search/dujianguang/lihaitao/qa_classify/lstm_code/data/wenwen/first_place_wenda/dumpres_wenda_result.qapair.new_model.out.click3'
        #self.test_out_file = self.test_file + '.mix.e9' 
        self.test_out_file = 'test.out'
        self.avilable_gpu = '4,5,6,7'
	    self.base_model = 'model_save_path/model_click3_2'
	    self.start = 3
	    self.start_from_base = False
	    self.istraining = True

def precision_recall(trueLabels, predictLabels):
  if not len(trueLabels) == len(predictLabels):
      return 0.0, 0.0, 0.0
  tp = 0.0
  tn = 0.0
  fp = 0.0
  fn = 0.0
  accuracyCount = 0.0
  for i in range(len(trueLabels)):
    if trueLabels[i] == predictLabels[i]:
        accuracyCount += 1
    if trueLabels[i] == 1 and predictLabels[i] == 1:
        tp = tp + 1
    elif trueLabels[i] == 1 and predictLabels[i] == 0:
        fn = fn + 1
    elif trueLabels[i] == 0 and predictLabels[i] == 1:
        fp = fp + 1
    elif trueLabels[i] == 0 and predictLabels[i] == 0:
        tn = tn + 1
  print("tp is %f, fp is %f, fn is %f, tn is %f"%(tp, fp, fn, tn))
  accuracy = accuracyCount / len(trueLabels)
  if tp == 0.0:
      return 0.0,0.0,accuracy 
  else:
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      return precision, recall, accuracy

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device

    return _assign
class Match_LSTM():
    def __init__(self, config):
        self.config = config
        #define placeholder
	batch_size = self.config.batch_size * self.config.num_gpus
        self.question_ph = tf.placeholder(tf.int32, [None, self.config.max_q_length], name="question_ph")
        self.answer_ph = tf.placeholder(tf.int32, [None, self.config.max_a_length], name="answer_ph")
        self.mask_q_ph = tf.placeholder(tf.float32, [None, self.config.max_q_length], name="mask_q_ph")
        self.mask_a_ph = tf.placeholder(tf.float32, [None, self.config.max_a_length], name="mask_a_ph")
        self.labels = tf.placeholder(tf.int32, [None, self.config.classes], name="label_ph")
        self.dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.w2v = word_embedding.Word2Vec(self.config.vocab_size, self.config.embedding_size)
        self.w2v.load_word_file(self.config.word_table_file)
        #self.question_embedding = tf.nn.embedding_lookup(self.w2v.id2embedding, self.question_ph)
        #self.answer_embedding = tf.nn.embedding_lookup(self.w2v.id2embedding, self.answer_ph)
        #model_out = self.build_model()
	    self.build_model_multi_gpu()
        self.build_tensor_table()
        #self.predict_probability = tf.nn.softmax(model_out)
        #self.predict_val = tf.argmax(model_out, 1)
        #correct_pred = tf.equal(self.predict_val, tf.argmax(self.labels, 1))
        #self.accuracy_val = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_out, labels=self.labels))
        #optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        #self.train_op = optimizer.minimize(self.loss)
        print 'init finished'
    def build_model_multi_gpu(self):
	    tower_grads = []
	    reuse_vars = False
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
	    for i in range(self.config.num_gpus):
	        with tf.device(assign_to_device('/gpu:%d'%i, ps_device='/cpu:0')):
                _q_ph = self.question_ph[i * self.config.batch_size : (i + 1) * self.config.batch_size]
                _a_ph = self.answer_ph[i * self.config.batch_size : (i + 1) * self.config.batch_size]
                _mask_q_ph = self.mask_q_ph[i * self.config.batch_size : (i + 1) * self.config.batch_size]
                _mask_a_ph = self.mask_a_ph[i * self.config.batch_size : (i + 1) * self.config.batch_size]
                question_embedding = tf.nn.embedding_lookup(self.w2v.id2embedding, _q_ph)
                answer_embedding = tf.nn.embedding_lookup(self.w2v.id2embedding, _a_ph)
		_labels_ph = self.labels[i * self.config.batch_size : (i + 1) * self.config.batch_size]
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

    def average_gradients(self, tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
	    grads = []
            for g, _ in grad_and_vars:
		expanded_g = tf.expand_dims(g, 0)
		grads.append(expanded_g)
	    grad = tf.concat(grads, 0)
	    grad = tf.reduce_mean(grad, 0)
	    v = grad_and_vars[0][1]
	    grad_and_var = (grad, v)
	    average_grads.append(grad_and_var)
	return average_grads


    def build_tensor_table(self):
        self.tensor_table = dict()
        #for v in tf.global_variables():
        for v in tf.trainable_variables():
            if re.match('embedding:0$', v.name):
                continue
            else:
		print v.name
                if v.name not in self.tensor_table.keys():
                    self.tensor_table[v.name] = v
    def bidirectional_lstm_layer(self, inputs, hidden_size, seq_length):
        forward_cell_unit = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=self.config.forget_bias)
        forward_cell_unit = tf.contrib.rnn.DropoutWrapper(forward_cell_unit, input_keep_prob=self.dropout_keep_prob_ph, output_keep_prob=self.dropout_keep_prob_ph)
        cell_unit = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=self.config.forget_bias)
        backward_cell_unit = tf.contrib.rnn.DropoutWrapper(cell_unit, input_keep_prob=self.dropout_keep_prob_ph, output_keep_prob=self.dropout_keep_prob_ph)
        outputs,_,_ = tf.contrib.rnn.static_bidirectional_rnn(forward_cell_unit, backward_cell_unit, inputs, dtype=tf.float32, sequence_length=seq_length)
	return outputs

    def lstm_layer(self, inputs, hidden_size, seq_length):
        cell_unit = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=self.config.forget_bias)
        cell_unit = tf.contrib.rnn.DropoutWrapper(cell_unit, input_keep_prob=self.dropout_keep_prob_ph, output_keep_prob=self.dropout_keep_prob_ph)
        outputs,_ = tf.contrib.rnn.static_rnn(cell_unit, inputs, dtype=tf.float32, sequence_length=seq_length)
        return outputs
    def word_by_word_attention(self, question, answer, softmax_q_mask):
        w_s = tf.get_variable(name="w_s", shape=[self.config.hidden_size * 2,self.config.hidden_size],
              dtype=tf.float32,
              initializer=tf.contrib.layers.xavier_initializer())
        w_t = tf.get_variable(name="w_t", shape=[self.config.hidden_size * 2,self.config.hidden_size],
              dtype=tf.float32,
              initializer=tf.contrib.layers.xavier_initializer())
        w_e = tf.get_variable(name="w_e", shape=[self.config.hidden_size, 1],
              dtype=tf.float32,
              initializer=tf.contrib.layers.xavier_initializer())
        a = [] 
        for k, output_a_ele in enumerate(answer):
            e = []
            for j, output_q_ele in enumerate(question):
                e_kj = tf.matmul(tf.tanh(tf.matmul(output_q_ele, w_s) + tf.matmul(output_a_ele, w_t)), w_e)
                e.append(e_kj)
	    e = tf.add(e, softmax_q_mask)
            alpha = tf.nn.softmax(tf.stack(e),0)
            a_k = tf.reduce_sum(tf.multiply(tf.stack(question), alpha), 0)
            a.append(a_k)
        return a
    def get_length(self, mask):
	tensor_length = tf.reduce_sum(mask,axis=-1)
	return tf.cast(tensor_length, tf.int32)

    def build_model(self, reuse_vars, question_embedding, answer_embedding, mask_q_ph, mask_a_ph):
        #format question and answer
        print 'in build model'
        question = tf.unstack(question_embedding, self.config.max_q_length, 1)
	q_length = self.get_length(mask_q_ph) - 1
	a_length = self.get_length(mask_a_ph)
        mask_q_list = tf.expand_dims(tf.transpose(mask_q_ph), -1)
	softmax_q_mask = tf.multiply((tf.subtract(1.0, mask_q_list)), float(-1e12))
	#softmax_q_mask = tf.multiply(mask_q_list, -1e12)
        with tf.variable_scope('question', reuse=reuse_vars):
            output_q = self.bidirectional_lstm_layer(question, self.config.hidden_size, q_length)
            #output_q = self.lstm_layer(question, self.config.hidden_size)
        output_q = tf.multiply(tf.stack(output_q), mask_q_list)
        output_q = tf.unstack(output_q, self.config.max_q_length, 0)

        answer = tf.unstack(answer_embedding, self.config.max_a_length, 1)
        mask_a_list = tf.expand_dims(tf.transpose(mask_a_ph), -1)
        with tf.variable_scope('question', reuse=True):
            output_a = self.bidirectional_lstm_layer(answer, self.config.hidden_size, a_length)
            #output_a = self.lstm_layer(answer, self.config.hidden_size)
        output_a = tf.multiply(tf.stack(output_a), mask_a_list)
        output_a = tf.unstack(output_a, self.config.max_a_length, 0)
	with tf.variable_scope("weights", reuse=reuse_vars):
            ww_attention = self.word_by_word_attention(output_q, output_a, softmax_q_mask)
        mLSTM_input = tf.concat([tf.stack(ww_attention), tf.stack(output_a)],2)
        #mLSTM_input = tf.nn.dropout(mLSTM_input, keep_prob=self.config.dropout_keep_prob)
        mLSTM_input = tf.unstack(mLSTM_input)
        with tf.variable_scope('final', reuse=reuse_vars):
            mLSTM_output = self.lstm_layer(mLSTM_input, self.config.hidden_size, a_length)
        #mLSTM_output = tf.nn.dropout(mLSTM_output, keep_prob=self.config.dropout_keep_prob)
        mLSTM_output_max = tf.reduce_max(tf.stack(mLSTM_output), axis=0)
        with tf.variable_scope('fully_connect', reuse=reuse_vars):
            final_output = tf.contrib.layers.fully_connected(inputs=mLSTM_output_max, num_outputs=self.config.classes, activation_fn=None)
        print 'build model finished'
        return final_output
    def train(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)
        print 'inited sess'
        sys.stdout.flush()
        self.w2v.load_tensorflow_embeddings(sess, self.config.word_embedding_file)
	start = 0
	if self.config.start_from_base:
	    print 'train the model base on the pretrained model'
            loader = tf.train.Saver(self.tensor_table.values())
            loader.restore(sess, self.config.base_model)
	    start = config.start
        for b in range(start,self.config.epoch):
	    self.config.num_gpus = len(self.config.gpus)
            start_time = time.time()
            train_ins = open(self.config.train_file, 'r')
            steps = 0
            loss =0
            acc = 0
            while True:
                success, q_sent_list, a_sent_list, target_list, mask_q_list, mask_a_list = self.w2v.get_intent_batch_mLSTM(train_ins, self.config.batch_size * self.config.num_gpus, self.config.max_q_length, self.config.max_a_length, self.config.classes)
                train_feed_dict = {
                        self.question_ph: q_sent_list,
                        self.answer_ph: a_sent_list,
                        self.labels: target_list,
                        self.mask_q_ph: mask_q_list,
                        self.mask_a_ph: mask_a_list,
                        self.dropout_keep_prob_ph: self.config.dropout_keep_prob
                        }
                _, train_loss = sess.run([self.train_op, self.loss], feed_dict=train_feed_dict)
                loss += float(train_loss)
                #acc += float(train_acc)
                steps += 1
                if steps % 1000 == 0:
		    train_feed_dict[self.dropout_keep_prob_ph] = 1.0
                    acc = sess.run(self.accuracy_val, feed_dict=train_feed_dict)
                    print('Epoch {0}, steps {1}: ave_loss = {2:.3f}, ave_accuracy={3:.3f}'.format(b, steps, loss / steps, acc))
                if not success:
	            train_feed_dict[self.dropout_keep_prob_ph] = 1.0
                    acc = sess.run(self.accuracy_val, feed_dict=train_feed_dict)
                    break
            end_time = time.time()
            print('Epoch {0} finished : ave_loss = {1:.3f}, ave_accuracy={2:.3f}'.format(b, loss / steps, acc))
	    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print 'time used {0} seconds'.format(str(end_time - start_time))

            saver = tf.train.Saver(self.tensor_table.values())
            saver.save(sess, self.config.model_save_file + '_' + str(b), write_meta_graph=True)
            self.eval_model(sess, self.config.eval_file)
            train_ins.close()
            sys.stdout.flush()

    def eval_model(self, sess, test_file):
        eval_ins = open(test_file, 'r')
        true_labels = []
        predict_labels = []
        predict_probability = []
        total_loss = .0
        step = 0
        while True:
            success, q_sent_list, a_sent_list, target_list, mask_q_list, mask_a_list = self.w2v.get_intent_batch_mLSTM(eval_ins, self.config.batch_size, self.config.max_q_length, self.config.max_a_length, self.config.classes)
            step += 1
            test_feed_dict = {
                    self.question_ph: q_sent_list,
                    self.answer_ph: a_sent_list,
                    self.labels: target_list,
                    self.mask_q_ph: mask_q_list,
                    self.mask_a_ph: mask_a_list,
                    self.dropout_keep_prob_ph: 1.0
                    }
            #test_acc, test_pre, test_pre_probability = sess.run([self.accuracy_val, self.predict_val, self.predict_probability], feed_dict=test_feed_dict)
            test_pre, test_pre_probability = sess.run([self.predict_val, self.predict_probability], feed_dict=test_feed_dict)
            #total_loss += test_loss
            true_labels.extend(np.argmax(target_list, 1))
            predict_labels.extend(test_pre)
            predict_probability.extend(test_pre_probability)
            if not success:
                break
        precision, recall, accuracy = precision_recall(true_labels, predict_labels)
        #print("Testing loss={0:.3f}" .format(total_loss / step))
        print("Testing accuracy={0:.3f}" .format(accuracy))
        print("Testing precision={0:.3f}" .format(precision))
        print("Testing recall={0:.3f}" .format(recall))

        eval_ins.close()
        sys.stdout.flush()
        return predict_probability
    def test_model(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)
        self.w2v.load_word_file(self.config.word_table_file)
        self.w2v.load_tensorflow_embeddings(sess, self.config.word_embedding_file)
        loader = tf.train.Saver(self.tensor_table.values())
        loader.restore(sess, self.config.model_save_file)
        print 'Model restored'
        predict_probability = self.eval_model(sess, self.config.test_file) 
        test_out_w = open(self.config.test_out_file, 'w')
        for i, ele in enumerate(predict_probability):
            test_out_w.write(str(ele[1]) + '\n')
        test_out_w.close()

if __name__ == '__main__':
    print 'main start'
    sys.stdout.flush()
    ops.reset_default_graph()
    config = CONFIG()
    if not config.istraining:
	print 'test model'
        config.gpus=['0']
	config.num_gpus=len(config.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.avilable_gpu 
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
        print 'start init mLSTM'
        mLSTM = Match_LSTM(config)
	if config.istraining:
            mLSTM.train(sess)    
	else:
            mLSTM.test_model(sess)
