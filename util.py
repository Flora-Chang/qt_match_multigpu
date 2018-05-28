# encoding: utf-8
import tensorflow as tf


flags = tf.app.flags

# Model parameters
flags.DEFINE_integer("filter_size", 64, "the num of filters of CNN")
flags.DEFINE_integer("embedding_size", 100, "words embedding size")
flags.DEFINE_integer("vocab_size", 4200000, "vocabulary size")
flags.DEFINE_float("keep_prob", 0.8, "dropout keep prob")

# change each runing
flags.DEFINE_string("flag", "word_3", "word/char/drmm")
flags.DEFINE_string("save_dir", "../logs/word_3_lr0.0005_bz128_poolsize80_1509519220", "save dir")
flags.DEFINE_string("predict_dir", "../output/result3", "predict result dir")

# Training / test parameters
flags.DEFINE_string("restore", False, "whether to restore old model")
flags.DEFINE_integer("query_len_threshold", 20, "threshold value of query length")
flags.DEFINE_integer("title_len_threshold", 20, "threshold value of document length")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("num_epochs", 10, "number of epochs")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_float("margin", 1.0, "cos margin")
#flags.DEFINE_integer("pooling_size", 80, "pooling size")

#File path
'''
flags.DEFINE_string("training_set", "../data/data_char/train.csv", "training set path")
flags.DEFINE_string("train_set", "../data/data_char/train.test.json", "train set path")
flags.DEFINE_string("dev_set", "../data/data_char/test.json", "dev set path")
flags.DEFINE_string("vocab_path", "../data/data_char/char_dict.txt", "vocab path")
flags.DEFINE_string("vectors_path", "../data/data_char/vectors_char.txt", "vectors path")
'''
flags.DEFINE_float("validation_steps", 1000, "steps between validations")
flags.DEFINE_float("GPU_rate", 0.5, "steps between validations")

flags.DEFINE_string("train_dir", "../data/train_data_new/train.txt", "training set path")
flags.DEFINE_string("tf_record_dir", "../data/tf_record/", "training set path")
flags.DEFINE_string("train_set", "../data/train.txt", "train set path")
flags.DEFINE_string("dev_set", "../data/dev.txt", "dev set path")
flags.DEFINE_string("vocab_path", "../data/word_table.tensorflow2", "vocab path")
flags.DEFINE_string("vectors_path", "../data/word_embedding.tensorflow", "vectors path")
flags.DEFINE_string("save_path", "../models/", "model save path")
flags.DEFINE_string("log_path", "../logs/", "log save path")
FLAGS = flags.FLAGS

