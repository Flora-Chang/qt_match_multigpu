#-*- coding:utf-8 -*-
#==================================================
#Anthor:Ruijun Wang
#Email:wangruijunSI3509@sogou-inc.com
#Copyright:2017 Sogou Inc. All Rights Reserved
#==================================================
import os
import re
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import time
import struct
import numpy as np
import pdb
import tensorflow as tf
#import io
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
class Word2Vec(object):
    def __init__(self, vocab_size, embedding_dim):
        self.word2id = {}
        #self.id2embedding = None
        self.id2word = []
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        with tf.device("/cpu:0"):
            self.id2embedding = tf.get_variable("embedding", [self.vocab_size+1, self.embedding_dim], dtype = np.float32,trainable=False)

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

    def get_wids(self, sent):
        words = sent.split(' ')
        wids = []
        for w in words:
            wids.append(self.get_wid(w))
        return wids

    def save_word_file(self, file_name):
        outs = open(file_name, 'w')
        for w in self.id2word:
            outs.write(w + '\n')
        outs.close()

    def load_word_file(self, file_name):
        sys.stderr.write('\nLoading word table...')
        ins = open(file_name, 'rb')
        line_num = 0
        while True:
            word = ins.readline()
            # print(word)
            if word == '':
                break
            word = word.rstrip(b'\n')

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
                line = ins.readline()
                #print "get_intent_batch"
                #f_out.write(str(line) + "\n")
            except Exception as e:
                print(Exception,":",e)
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
                         qa_match1_return[i] =  qa_match1[i]
                         qa_match2_return[i] = qa_match2[i]
                    a_sent_list_return = np.array([a_sent_list1_return, a_sent_list2_return])
                    qa_match_return = np.array([qa_match1_return, qa_match2_return])
                    return True, q_sent_list_return, a_sent_list_return
                    #return True, q_sent_list, a_sent_list, target_list

            line = line.rstrip('\n')
            # print (line)
            fields = line.split('\t')
            if len(fields) != 3:
                raise ValueError("Invalid line: %s, which should have 3 fields.", line)
            q_wids = self.get_wids(fields[0])
            a_wids1 = self.get_wids(fields[1])
            a_wids2 = self.get_wids(fields[2])
            #score = fields[2]
            # wids.insert(0, 0)
            # scores.insert(0, '0.0')
            q_wids.append(0)
            a_wids1.append(0)
            a_wids2.append(0)
            # if len(wids) != len(scores):
            # raise ValueError("Invalid line: %s, different length between sentence and target.", line)

            if len(q_wids) > Qlen:
                q_wids = q_wids[:Qlen]
            if len(a_wids1) > Alen:
                a_wids1 = a_wids1[:Alen]
            if len(a_wids2) > Alen:
                a_wids2 = a_wids2[:Alen]

            # 后面补0
            for i in range(len(q_wids)):
                q_sent_list[batch_cursor, i] = q_wids[i]
            #for i in range(len(q_wids)):
            #    if i == len(q_wids) - 1:
            #        target_list[batch_cursor] = float(score)
            
            #target_list[batch_cursor] = float(score)
            for j in range(len(a_wids1)):
                a_sent_list1[batch_cursor, j] = a_wids1[j]
            for j in range(len(a_wids2)):
                a_sent_list2[batch_cursor, j] = a_wids2[j]
            a_sent_list = [a_sent_list1, a_sent_list2]
            for i in range(len(q_wids)):
                for j in range(len(a_wids1)):
                    if q_wids[i] == a_wids1[j]:
                        qa_match1[batch_cursor, i, j] = 1
                    if q_wids[i] == a_wids2[j]:
                        qa_match2[batch_cursor, i, j] = 1
            qa_match = np.array([qa_match1, qa_match2])

            batch_cursor += 1
            if batch_cursor == batch_size:
                #print (sent_list)
                return True, q_sent_list, a_sent_list, qa_match

    def get_intent_batch_mLSTM(self, ins, batch_size, q_num_steps, a_num_steps, n_class):
        batch_cursor = 0
        q_sent_list = np.zeros([batch_size, q_num_steps], dtype=np.int32)
        a_sent_list = np.zeros([batch_size, a_num_steps], dtype=np.int32)
        target_list = np.zeros([batch_size, n_class], dtype=np.int32)
        length_q_list = np.zeros([batch_size, q_num_steps], dtype=np.float32)
        length_a_list = np.zeros([batch_size, a_num_steps], dtype=np.float32)
        #frameweight_list = np.zeros([num_steps, batch_size], dtype=np.float32)

        # sys.stderr.write('\nLoading training corpus ...\n')
        while True:
            try:
                #line = ins.readline().decode("utf8").encode("gbk")
                line = ins.readline()
                #print "get_intent_batch"
                #f_out.write(str(line) + "\n")
            except Exception as e:
                print(Exception,":",e)
            if line == '':
                if batch_cursor == 0:
                    return False, q_sent_list, a_sent_list, target_list, length_q_list, length_a_list
                else:
                    q_sent_list_return = np.zeros([batch_cursor, q_num_steps], dtype=np.int32)
                    a_sent_list_return = np.zeros([batch_cursor, a_num_steps], dtype=np.int32)
                    target_list_return = np.zeros([batch_cursor, n_class], dtype=np.int32)
                    length_q_list_return = np.zeros([batch_cursor, q_num_steps], dtype=np.float32)
                    length_a_list_return = np.zeros([batch_cursor, a_num_steps], dtype=np.float32)
                    for i in range(batch_cursor):
                         q_sent_list_return[i] = q_sent_list[i]
                         a_sent_list_return[i] = a_sent_list[i]
                         target_list_return[i] = target_list[i]
                         length_q_list_return[i] = length_q_list[i]
                         length_a_list_return[i] = length_a_list[i]
                    return True, q_sent_list_return, a_sent_list_return, target_list_return, length_q_list_return, length_a_list_return

            line = line.rstrip('\n')
            #print line.decode('gbk', 'ignore').encode('utf-8', 'ignore')
            fields = line.split('\t')
            if len(fields) != 3:
                raise ValueError("Invalid line: %s, which should have 3 fields.", line)
            q_wids = self.get_wids(fields[0])
            a_wids = self.get_wids(fields[1])
            score = fields[2]
            # wids.insert(0, 0)
            # scores.insert(0, '0.0')
            #q_wids.append(0)
            #a_wids.append(0)
            # if len(wids) != len(scores):
            # raise ValueError("Invalid line: %s, different length between sentence and target.", line)

            if len(q_wids) > q_num_steps:
                q_wids = q_wids[:q_num_steps]
            if len(a_wids) > a_num_steps:
                a_wids = a_wids[:a_num_steps]

            # 后面补0
            for i in range(len(q_wids)):
                q_sent_list[batch_cursor, i] = q_wids[i]
                length_q_list[batch_cursor][i] = 1 
            #for i in range(len(q_wids)):
            #    if i == len(q_wids) - 1:
            #        target_list[batch_cursor] = float(score)
            target_list[batch_cursor][int(score)] = 1 
            for j in range(len(a_wids)):
                a_sent_list[batch_cursor, j] = a_wids[j]
                length_a_list[batch_cursor][j] = 1
            #mask matrix
            #length_list[batch_cursor] = len(wids)
            #for i in range(len(wids)):
                #if i == len(wids) - 1:
                    #frameweight_list[i, batch_cursor] = 1.0

            #############################################################
            # 前面补0
            # for i in range(len(wids)):
            #   sent_list[batch_cursor, num_steps-len(wids)+i] = wids[i]
            # target_list[num_steps-1,batch_size] = float(score)
            # length_list[batch_cursor] = num_steps
            # frameweight_list[num_steps-1,batch_cursor] = 1.0
            ###############################################################

            batch_cursor += 1
            if batch_cursor == batch_size:
                #print batch_cursor
                return True, q_sent_list, a_sent_list, target_list, length_q_list, length_a_list
