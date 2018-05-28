from model_old import *
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=config) as sess:
    model = Model(mode='eval')
    #model.eval(sess=sess, in_train=False, data_mode="test")
    model.eval_qtitle(sess, data_mode="test")