from model_old import *
#from model_tfrecord import *
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=config) as sess:
    model = Model(mode="train")
    model.train(sess=sess)