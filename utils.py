def initTf(tf, K):
    #Tensorflow GPU optimization
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    sess = tf.Session()
    K.set_session(sess)