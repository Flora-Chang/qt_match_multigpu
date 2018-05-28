from word_embedding import *
from util import FLAGS
import os

w2v = Word2Vec(FLAGS.vocab_size, FLAGS.embedding_size)
w2v.load_word_file(FLAGS.vocab_path)
train_dir = FLAGS.train_dir
tf_record_dir = FLAGS.tf_record_dir
if not os.path.exists(tf_record_dir):
    os.makedirs(tf_record_dir)
for file in list(os.listdir(train_dir)):
    w2v.build_tfrecord(train_dir+file,tf_record_dir+file, Qlen=20, Alen=20)

