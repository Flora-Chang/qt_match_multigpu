# -*- coding:utf-8 -*-
# ==================================================
# Anthor:Ruijun Wang
# Email:wangruijunSI3509@sogou-inc.com
# Copyright:2017 Sogou Inc. All Rights Reserved
# ==================================================
import os
import re
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import time
import struct
import numpy as np
import pdb
import tensorflow as tf
# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
def strQ2B(ustring):
    '''全转半'''
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


class Word2Vec(object):
    def __init__(self, vocab_size, embedding_dim):
        self.word2id = {}
        # self.id2embedding = None
        self.id2word = []
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        with tf.device("/cpu:0"):
            self.id2embedding = tf.get_variable("embedding", [self.vocab_size + 1, self.embedding_dim],
                                                dtype=np.float32, trainable=False)

    def get_word(self, wid):
        assert wid <= self.vocab_size
        return self.id2word[wid]

    def get_wid(self, word):
        # print(self.word2id.keys())
        # print(word)
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.vocab_size

    def get_wids(self, sent, sentLen):
        #words = sent.split(' ')
        words = list(sent)
        wids = [0] * sentLen
        if len(words) > sentLen:
            words = words[:sentLen]
        for i, w in enumerate(words):
            wids[i] = self.get_wid(w)
        return wids

    def save_word_file(self, file_name):
        outs = open(file_name, 'w')
        for w in self.id2word:
            outs.write(w + '\n')
        outs.close()

    def load_word_file(self, file_name):
        sys.stderr.write('\nLoading word table...')
        #count = 0
        ins = open(file_name, 'r')
        line_num = 0
        while True:
            word = ins.readline()
            word = word.strip('\n')
            word = strQ2B(word)
            #if count % 10000 == 0:
                #print("count: ", count, word)
            #count += 1
            if word == '':
                break
            self.id2word.append(word)
            self.word2id[word] = line_num
            line_num += 1
        ins.close()
        sys.stderr.write('\rLoading word table finished !')

    def save_tensorflow_embeddings(self, sess, file_name):
        saver = tf.train.Saver()
        saver.save(sess, file_name)

    def load_tensorflow_embeddings(self, sess, file_name):
        sys.stderr.write('\nLoading word embeddings from tensor file...\n')
        saver = tf.train.Saver({'embedding': self.id2embedding})
        saver.restore(sess, file_name)
        sys.stderr.write('\nLoading word embeddings finished !\n')

    def get_intent_batch(self, ins, batch_size, Qlen, Alen):
        batch_cursor = 0
        q_sent_list = np.zeros([batch_size, Qlen], dtype=np.int32)
        a_sent_list1 = np.zeros([batch_size, Alen], dtype=np.int32)
        a_sent_list2 = np.zeros([batch_size, Alen], dtype=np.int32)
        qa_match1 = np.zeros([batch_size, Qlen, Alen], dtype=np.float32)
        qa_match2 = np.zeros([batch_size, Qlen, Alen], dtype=np.float32)

        # sys.stderr.write('\nLoading training corpus ...\n')
        while True:
            try:
                line = ins.readline().strip()
                # print "get_intent_batch"
                # f_out.write(str(line) + "\n")
            except Exception as e:
                print(Exception, ":", e)
            if line == '':
                if batch_cursor == 0:
                    a_sent_list = np.array([a_sent_list1, a_sent_list2])
                    qa_match = np.array([qa_match1, qa_match2])
                    return False, q_sent_list, a_sent_list, qa_match
                else:
                    q_sent_list_return = np.zeros([batch_cursor, Qlen], dtype=np.int32)
                    a_sent_list1_return = np.zeros([batch_cursor, Alen], dtype=np.int32)
                    a_sent_list2_return = np.zeros([batch_cursor, Alen], dtype=np.int32)
                    qa_match1_return = np.zeros([batch_cursor, Qlen, Alen], dtype=np.float32)
                    qa_match2_return = np.zeros([batch_cursor, Qlen, Alen], dtype=np.float32)
                    for i in range(batch_cursor):
                        q_sent_list_return[i] = q_sent_list[i]
                        a_sent_list1_return[i] = a_sent_list1[i]
                        a_sent_list2_return[i] = a_sent_list2[i]
                        qa_match1_return[i] = qa_match1[i]
                        qa_match2_return[i] = qa_match2[i]
                    a_sent_list_return = np.array([a_sent_list1_return, a_sent_list2_return])
                    qa_match_return = np.array([qa_match1_return, qa_match2_return])
                    return True, q_sent_list_return, a_sent_list_return, qa_match_return
                    # return True, q_sent_list, a_sent_list, target_list

            line = line.rstrip('\n')
            # print (line)
            fields = line.split('\t')
            if len(fields) != 3:
                raise ValueError("Invalid line: %s, which should have 3 fields.", line)
            q_wids = self.get_wids(fields[0], sentLen=Qlen)
            a_wids1 = self.get_wids(fields[1], sentLen=Alen)
            a_wids2 = self.get_wids(fields[2], sentLen=Alen)

            for i in range(len(q_wids)):
                q_sent_list[batch_cursor, i] = q_wids[i]
            # for i in range(len(q_wids)):
            #    if i == len(q_wids) - 1:
            #        target_list[batch_cursor] = float(score)

            # target_list[batch_cursor] = float(score)
            for j in range(len(a_wids1)):
                a_sent_list1[batch_cursor, j] = a_wids1[j]
            for j in range(len(a_wids2)):
                a_sent_list2[batch_cursor, j] = a_wids2[j]
            a_sent_list = [a_sent_list1, a_sent_list2]
            for i in range(len(q_wids)):
                for j in range(len(a_wids1)):
                    if q_wids[i] == a_wids1[j]:
                        qa_match1[batch_cursor, i, j] = 1
                for j in range(len(a_wids2)):
                    if q_wids[i] == a_wids2[j]:
                        qa_match2[batch_cursor, i, j] = 1
            qa_match = np.array([qa_match1, qa_match2])

            batch_cursor += 1
            if batch_cursor == batch_size:
                # print (sent_list)
                return True, q_sent_list, a_sent_list, qa_match

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def build_tfrecord(self, data_path, tfrecords_path, Qlen, Alen):
        self.Qlen = Qlen
        self.Alen = Alen
        writer = tf.python_io.TFRecordWriter(tfrecords_path)
        with open(data_path, 'r') as train_f:
            for line in train_f:
                line = line.strip().split('\t')
                if len(line) != 3:
                    print(line)
                    continue
                ori_query = line[0].strip().split()
                ori_title1 = line[1].strip().split()
                ori_title2 = line[2].strip().split()
                query = self.get_wids(ori_query, sentLen=Qlen)
                title_1 = self.get_wids(ori_title1, sentLen=Alen)
                title_2 = self.get_wids(ori_title2, sentLen=Alen)
                '''
                titles = np.array([title_1, title_2]).tostring()
                pos_qt_match = np.zeros((Qlen, Alen), dtype=int)
                neg_qt_match = np.zeros((Qlen, Alen), dtype=int)
                for i in range(min([len(ori_query), Qlen])):
                    for j in range(min(len(ori_title1), Alen)):
                        if ori_query[i] == ori_title1[j]:
                            pos_qt_match[i, j] = 1
                for i in range(min([len(ori_query), Qlen])):
                    for j in range(min(len(ori_title2), Alen)):
                        if ori_query[i] == ori_title2[j]:
                            neg_qt_match[i, j] = 1
                qt_match = np.array([pos_qt_match, neg_qt_match]).tostring()
                

                example = tf.train.Example(features=tf.train.Features(feature={
                    'query': self._int64_feature(query),
                    'titles':  self._bytes_feature(titles),
                    'qt_match': self._bytes_feature(qt_match)}
                ))
                '''
                example = tf.train.Example(features=tf.train.Features(feature={
                    'query': self._int64_feature(query),
                    'pos_title': self._int64_feature(title_1),
                    'neg_title': self._int64_feature(title_2)}
                ))
                writer.write(example.SerializeToString())
        writer.close()

    def load_tfrecord(self, tfrecord_paths):
        # Even when reading in multiple threads, share the filename
        # queue.
        Qlen = 20
        Alen = 20
        filename_queue = tf.train.string_input_producer(tfrecord_paths)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'query': tf.FixedLenFeature(Qlen, tf.int64),
                                               'pos_title': tf.FixedLenFeature(Alen, tf.int64),
                                               'neg_title': tf.FixedLenFeature(Alen, tf.int64),
                                           })
        query = features['query']
        pos_title = features['pos_title']
        neg_title = features['neg_title']
        print(query)
        print(pos_title)
        print(neg_title)

        query_, pos_title_, neg_title_ = tf.train.shuffle_batch(
            [query, pos_title, neg_title],
            batch_size=128,
            capacity=10000,
            num_threads=5,
            enqueue_many=False,
            min_after_dequeue=5000)

        return  query_, pos_title_, neg_title_
    '''
    def build_feature_local(self, query, title, max_query_word, max_title_word):
        query_len = min([max_query_word, len(query)])
        title_len = min([max_title_word, len(title)])
        qa_match = np.zeros((max_query_word, max_title_word))
        for i, word_q in enumerate(query[:query_len]):
            for j, word_t in enumerate(title[:title_len]):
                if word_q == word_t:
                    qa_match[i,j] = 1
        return qa_match
    '''

    def get_train_batch(self, file, batch_size, Qlen=20, Alen=20):
        index = 0
        data = open(file, 'r').readlines()
        data_size = len(data)
        #shuffle_indices = np.random.permutation(np.arange(data_size))
        #shuf_data = data[shuffle_indices]
        np.random.shuffle(data)
        while (index) * batch_size < data_size:
            if (index + 1) * batch_size <= data_size:
                batch_data =  data[index * batch_size: (index + 1) * batch_size]
            else:
                batch_data = data[index * batch_size: data_size]
            index += 1
            queries = []
            pos_titles = []
            neg_titles = []
            qt_matchs = []
            titles = []

            for line in batch_data:
                line = line.strip().split('\t')
                local_query = list(strQ2B(line[0]).lower().strip().split())
                local_title1 = list(strQ2B(line[1]).lower().strip().split())
                local_title2 = list(strQ2B(line[2]).lower().strip().split())
                pos_qt_match = self.build_feature_local(local_query, local_title1, Qlen, Alen)
                neg_qt_match = self.build_feature_local(local_query, local_title2, Qlen, Alen)
                qt_match = [pos_qt_match, neg_qt_match]
                query = self.get_wids(local_query, Qlen)
                pos_title = self.get_wids(local_title1, Alen)
                neg_title = self.get_wids(local_title2, Alen)
                queries.append(query)
                pos_titles.append(pos_title)
                neg_titles.append(neg_title)
                qt_matchs.append(qt_match)
            for pos, neg in zip(pos_titles, neg_titles):
                titles.append([pos, neg])
            yield queries, titles, qt_matchs

    def get_eval_batch(self, eval_path, batch_size, Qlen, Alen):
        index = 0
        cnt = 0
        data = open(eval_path, 'r').readlines()
        data_size = len(data)

        while (index) * batch_size < data_size:
            if (index + 1) * batch_size <= data_size:
                batch_data = data[index * batch_size: (index + 1) * batch_size]
            else:
                batch_data = data[index * batch_size: data_size]
            index += 1
            queries = []
            pos_titles = []
            neg_titles = []
            qt_matchs = []
            titles = []

            for line in batch_data:
                line = line.strip().split('\t')
                local_query = list(strQ2B(line[0]).lower().strip().split())
                local_title1 = list(strQ2B(line[1]).lower().strip().split())
                local_title2 = list(strQ2B(line[2]).lower().strip().split())
                pos_qt_match = self.build_feature_local(local_query, local_title1, Qlen, Alen)
                neg_qt_match = self.build_feature_local(local_query, local_title2, Qlen, Alen)
                qt_match = [pos_qt_match, neg_qt_match]
                query = self.get_wids(local_query, Qlen)
                pos_title = self.get_wids(local_title1, Alen)
                neg_title = self.get_wids(local_title2, Alen)
                queries.append(query)
                pos_titles.append(pos_title)
                neg_titles.append(neg_title)
                qt_matchs.append(qt_match)
            for pos, neg in zip(pos_titles, neg_titles):
                titles.append([pos, neg])
            yield queries, titles, qt_matchs

        print("self.cnt:", cnt)

    def get_qtitle10w_batch(self, data_path, batch_size, Qlen, Alen):
        index = 0
        cnt = 0
        data = open(data_path, 'r').readlines()
        data_size = len(data)

        while (index) * batch_size < data_size:
            if (index + 1) * batch_size <= data_size:
                batch_data = data[index * batch_size: (index + 1) * batch_size]
            else:
                batch_data = data[index * batch_size: data_size]
            index += 1
            queries = []
            titles = []
            labels = []
            qt_matchs = []
            indexes = []
            new_titles = []

            for line in batch_data:
                cnt += 1
                line = line.strip().split('\t')
                local_query = list(strQ2B(line[0]).lower().strip().split())
                local_title1 = list(strQ2B(line[1]).lower().strip().split())
                label = line[2].strip()
                i = line[3].strip()
                query = self.get_wids(local_query, Qlen)
                title = self.get_wids(local_title1, Alen)
                qt_match = self.build_feature_local(local_query, local_title1, Qlen, Alen)
                queries.append(query)
                #titles.append(title)
                titles.append([title, title])
                labels.append(label)
                indexes.append(i)
                qt_matchs.append([qt_match, qt_match])
            yield queries, titles, qt_matchs, labels, indexes

        print("self.cnt:", cnt)


