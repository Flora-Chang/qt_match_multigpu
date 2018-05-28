#coding=utf-8
import tensorflow as tf
import numpy as np
import word_embedding as word_embedding
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
        self.max_q_length = 20
        self.max_a_length = 200
        self.learning_rate = 0.001
        self.hidden_size = 300
        self.dropout_keep_prob = 0.7
        self.classes = 2
        self.batch_size = 128
        self.epoch = 10
        self.forget_bias = 1.0
        self.word_table_file = '/search/odin/dujianguang/word_table.tensorflow'
        self.word_embedding_file = '/search/odin/dujianguang/word_embedding.tensorflow'
        self.train_file = '../data/ccir/ccir.train.all.full.new'
        self.eval_file = '../data/ccir/ccir.test.long'
        self.test_file = ''
        self.model_save_file = 'model_save_path/test'
        self.test_out_file = 'test.out'
        self.avilable_gpu = '7'

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

class Match_LSTM():
    def __init__(self, config):
        self.config = config
        #define placeholder
        self.question_ph = tf.placeholder(tf.int32, [None, self.config.max_q_length], name="question_ph")
        self.answer_ph = tf.placeholder(tf.int32, [None, self.config.max_a_length], name="answer_ph")
        self.mask_q_ph = tf.placeholder(tf.float32, [None, self.config.max_q_length], name="mask_q_ph")
        self.mask_a_ph = tf.placeholder(tf.float32, [None, self.config.max_a_length], name="mask_a_ph")
        self.labels = tf.placeholder(tf.int32, [None, self.config.classes], name="label_ph")
        self.dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.w2v = word_embedding.Word2Vec(self.config.vocab_size, self.config.embedding_size)
        self.question_embedding = tf.nn.embedding_lookup(self.w2v.id2embedding, self.question_ph)
        self.answer_embedding = tf.nn.embedding_lookup(self.w2v.id2embedding, self.answer_ph)
        model_out = self.build_model()
        self.build_tensor_table()
        self.predict_probability = tf.nn.softmax(model_out)
        self.predict_val = tf.argmax(model_out, 1)
        correct_pred = tf.equal(self.predict_val, tf.argmax(self.labels, 1))
        self.accuracy_val = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_out, labels=self.labels))
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        print 'init finished'
    def build_tensor_table(self):
        self.tensor_table = dict()
        for v in tf.global_variables():
            if re.match('embedding:0$', v.name):
                continue
            else:
                if v.name not in self.tensor_table.keys():
                    self.tensor_table[v.name] = v

    def lstm_layer(self, inputs, hidden_size):
        cell_unit = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=self.config.forget_bias)
        cell_unit = tf.contrib.rnn.DropoutWrapper(cell_unit, input_keep_prob=self.config.dropout_keep_prob, output_keep_prob=self.config.dropout_keep_prob)
        outputs,_ = tf.contrib.rnn.static_rnn(cell_unit, inputs, dtype=tf.float32)
        return outputs
    def word_by_word_attention(self, question, answer):
        w_s = tf.get_variable(name="w_s", shape=[self.config.hidden_size,self.config.hidden_size],
              dtype=tf.float32,
              initializer=tf.contrib.layers.xavier_initializer())
        w_t = tf.get_variable(name="w_t", shape=[self.config.hidden_size,self.config.hidden_size],
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
            alpha = tf.nn.softmax(tf.stack(e),0)
            a_k = tf.reduce_sum(tf.multiply(tf.stack(question), alpha), 0)
            a.append(a_k)
        return a

    def build_model(self):
        #format question and answer
        print 'in build model'
        question = tf.unstack(self.question_embedding, self.config.max_q_length, 1)
        mask_q_list = tf.expand_dims(tf.transpose(self.mask_q_ph), -1)
        with tf.variable_scope('question'):
            output_q = self.lstm_layer(question, self.config.hidden_size)
        output_q = tf.multiply(tf.stack(output_q), mask_q_list)
        output_q = tf.unstack(output_q, self.config.max_q_length, 0)

        answer = tf.unstack(self.answer_embedding, self.config.max_a_length, 1)
        mask_a_list = tf.expand_dims(tf.transpose(self.mask_a_ph), -1)
        with tf.variable_scope('question', reuse=True):
            output_a = self.lstm_layer(answer, self.config.hidden_size)
        output_a = tf.multiply(tf.stack(output_a), mask_a_list)
        output_a = tf.unstack(output_a, self.config.max_a_length, 0)
        ww_attention = self.word_by_word_attention(output_q, output_a)
        mLSTM_input = tf.concat([tf.stack(ww_attention), tf.stack(output_a)],2)
        #mLSTM_input = tf.nn.dropout(mLSTM_input, keep_prob=self.config.dropout_keep_prob)
        mLSTM_input = tf.unstack(mLSTM_input)
        with tf.variable_scope('final'):
            mLSTM_output = self.lstm_layer(mLSTM_input, self.config.hidden_size)
        #mLSTM_output = tf.nn.dropout(mLSTM_output, keep_prob=self.config.dropout_keep_prob)
        mLSTM_output_max = tf.reduce_max(tf.stack(mLSTM_output), axis=0)
        final_output = tf.contrib.layers.fully_connected(inputs=mLSTM_output_max, num_outputs=self.config.classes, activation_fn=None)
        print 'build model finished'
        return final_output
    def train(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)
        print 'inited sess'
        sys.stdout.flush()
        self.w2v.load_word_file(self.config.word_table_file)
        self.w2v.load_tensorflow_embeddings(sess, self.config.word_embedding_file)
        for b in range(self.config.epoch):
            start_time = time.time()
            train_ins = open(self.config.train_file, 'r')
            steps = 0
            loss =0
            acc = 0
            while True:
                success, q_sent_list, a_sent_list, target_list, mask_q_list, mask_a_list = self.w2v.get_intent_batch_mLSTM(train_ins, self.config.batch_size, self.config.max_q_length, self.config.max_a_length, self.config.classes)
                if not success:
                    break
                train_feed_dict = {
                        self.question_ph: q_sent_list,
                        self.answer_ph: a_sent_list,
                        self.labels: target_list,
                        self.mask_q_ph: mask_q_list,
                        self.mask_a_ph: mask_a_list,
                        self.dropout_keep_prob_ph: self.config.dropout_keep_prob
                        }
                _, train_loss, train_acc = sess.run([self.train_op, self.loss, self.accuracy_val], feed_dict=train_feed_dict)
                loss += float(train_loss)
                acc += float(train_acc)
                steps += 1
                if steps % 100 == 0:
                    print('Epoch {0}, steps {1}: ave_loss = {2:.3f}, ave_accuracy={3:.3f}'.format(b, steps, loss / steps, acc / steps))
            end_time = time.time()
            print('Epoch {0} finished : ave_loss = {1:.3f}, ave_accuracy={2:.3f}'.format(b, loss / steps, acc / steps))
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
            if not success:
                break
            step += 1
            test_feed_dict = {
                    self.question_ph: q_sent_list,
                    self.answer_ph: a_sent_list,
                    self.labels: target_list,
                    self.mask_q_ph: mask_q_list,
                    self.mask_a_ph: mask_a_list,
                    self.dropout_keep_prob_ph: 1.0
                    }
            test_acc, test_pre, test_loss, test_pre_probability = sess.run([self.accuracy_val, self.predict_val, self.loss, self.predict_probability], feed_dict=test_feed_dict)
            total_loss += test_loss
            true_labels.extend(np.argmax(target_list, 1))
            predict_labels.extend(test_pre)
            predict_probability.extend(test_pre_probability)
        precision, recall, accuracy = precision_recall(true_labels, predict_labels)
        print("Testing loss={0:.3f}" .format(total_loss / step))
        print("Testing accuracy={0:.3f}" .format(accuracy))
        print("Testing precision={0:.3f}" .format(precision))
        print("Testing recall={0:.3f}" .format(recall))

        eval_ins.close()
        sys.stdout.flush()
        return test_pre_probability
    def test_model(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)
        self.w2v.load_word_file(self.config.word_table_file)
        self.w2v.load_tensorflow_embeddings(sess, self.config.word_embedding_file)
        loader = tf.train.Saver(self.tensor_table.values())
        loader.restor(sess, self.config.model_save_file)
        print 'Model restored'
        predict_probability = eval_model(sess, self.config.test_file) 
        test_out_w = open(self.config.test_out_file, 'w')
        for i, ele in enumerate(predict_probability):
            test_out_w.write(str(ele[1]) + '\n')
        test_out_w.close()

if __name__ == '__main__':
    print 'main start'
    sys.stdout.flush()
    ops.reset_default_graph()
    config = CONFIG()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.avilable_gpu 
    with tf.Session() as sess:
        print 'start init mLSTM'
        mLSTM = Match_LSTM(config)
        mLSTM.train(sess)


def train(self, sess):
    labels_start = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size])
    labels_end = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size])
    label = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size])

    logits = tf.concat(values=[self.score_start, self.score_end], axis=0)
    labels = tf.concat(values=[labels_start, labels_end], axis=0)
    new_label = tf.concat(values=[label, label], axis=0)

    print('build loss')
    self.loss1 = tf.reduce_mean(tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels), tf.cast(new_label, dtype=tf.float64)))
    self.loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score_label, labels=label))
    self.loss = self.loss1 + 0.8 * self.loss2
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config.learning_rate, rho=0.95,
                                           epsilon=1e-6)  # .minimize(loss=self.loss)
    print('built loss')

    grad_and_vars = optimizer.compute_gradients(loss=self.loss)

    for grad, var in grad_and_vars:
        tf.summary.histogram(name='grad_' + var.name, values=grad)

    opt = optimizer.apply_gradients(grads_and_vars=grad_and_vars)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir=self.config.log_path)
    #dataloader = DataLoader(self.config.data_path, batch_size=self.config.batch_size, max_len=self.config.Plen)
    dataloader = DataLoad(vocab_dict=self.vocab_id_dict, data_path=self.config.data_path, batch_size=self.config.batch_size, data_mode="train")
    print('data loaded')
    tf.global_variables_initializer().run()
    print('variables initialized')

    self.saver = tf.train.Saver()
    if self.config.restore:
        self.saver.restore(sess=sess, save_path=self.config.save_path)
    epoch = 0
    loss_sum = 0.
    steps = 0
    match_count = 0.
    f1_before = 0
    while (epoch < self.config.epochs):
        epoch += 1
        for batch_data in dataloader.next_batch():
            steps += 1
            #context_list, question_list, start_label_list, end_label_list, _, epoch = batch_data
            (query_ids, queries), (passage_ids, passages), (label_list, start_list, end_list) = batch_data
            P_ids_list = self.get_ids_list(passages, self.config.Plen)
            Q_ids_list = self.get_ids_list(queries, self.config.Qlen)
            _, loss, pos_list, true_label, summary_str = sess.run([opt, self.loss, self.pos, self.true_label, summary],
                                                      feed_dict={label: label_list,
                                                                 labels_start: start_list,
                                                                 labels_end: end_list,
                                                                 self.P_ids: P_ids_list,
                                                                 self.Q_ids: Q_ids_list})
            loss_sum += loss
            summary_writer.add_summary(summary=summary_str, global_step=steps)
            for i in range(self.config.batch_size):
                if start_list[i] == pos_list[i][0] and end_list[i] == pos_list[i][1] and true_label[i]==1:
                    match_count += 1
            if steps % 20 == 0:
                s = u' '
                context = []
                question = ''
                for i in P_ids_list[0]:
                    context.append(self.id_vocab_dict[i])
                for i in Q_ids_list[0]:
                    question += self.id_vocab_dict[i] + ' '
                pos = pos_list.tolist()[0]
                start = start_list[0]
                end = end_list[0]
                tmp_label = label_list[0]
                print("exact_match: ", match_count / (self.config.batch_size * 20))
                match_count = 0.
                print(self.config.model_name)
                print(context)
                print(passages[0])
                print(question)
                print([start, end, tmp_label])
                print(pos)
                print('Ground Truth: ', context[start:end + 1], "laebl: ", tmp_label)
                try:
                    print('Eval Answer: ', context[pos[0]:(pos[1] + 1)], "label: ",true_label[0] )
                except Exception as e:
                    print('Out of context range!')
                print('loss: ', loss_sum / 20)
                print('epoch: ', epoch)
                print('steps: ', steps)
                print('=========================\n')
                print("on validation set: ")
                self.saver.save(sess=sess, save_path=self.config.save_path)
                self.mode = 'eval'
                self.eval(sess=sess, data_mode='valid')
                exact_match, char_match, f1 = computeScore(self.config.result_path_qidans)
                if f1 > f1_before:
                    self.saver.save(sess=sess, save_path=self.config.best_save_path)
                    f1_before = f1
                    print("saved to best models")
                print("=============================")
                print("exact_match: ", exact_match)
                print("char_match: ", char_match)
                print("f1: ", f1)
                print("=============================")
                self.mode = 'train'
                loss_sum = 0.

