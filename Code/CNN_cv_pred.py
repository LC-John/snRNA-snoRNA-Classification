# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 09:35:01 2017

@author: DrLC
"""

import tensorflow as tf
from dataset import load_data
from dataset import Dataset
import numpy

test_file = "../Dataset/final_m.pkl.gz"
k_fold = 10

tf.set_random_seed(0)

X = tf.placeholder(tf.float32, [None, 60, 4])
Y_ = tf.placeholder(tf.float32, [None, 2])
lr = tf.placeholder(tf.float32)
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()
    
def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

# Convolutional kernel
K = 16
L = 32
M = 64
W1 = tf.Variable(tf.truncated_normal([1, 4, 1, K], stddev=0.1), name="W1")
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]), name="B1")
W2 = tf.Variable(tf.truncated_normal([10, 1, K, L], stddev=0.1), name="W2")
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]), name="B2")
W3 = tf.Variable(tf.truncated_normal([10, 1, L, M], stddev=0.1), name="W3")
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]), name="B3")

# Full connection
H = 128
W4 = tf.Variable(tf.truncated_normal([6*1*M, H], stddev=0.1), name="W4")
B4 = tf.Variable(tf.constant(0.1, tf.float32, [H]), name="B4")
W5 = tf.Variable(tf.truncated_normal([H, 2], stddev=0.1), name="W5")
B5 = tf.Variable(tf.constant(0.1, tf.float32, [2]), name="B5")


stride = 1

XX = tf.reshape(X, shape=[-1, 60, 4, 1])
Y1l = tf.nn.conv2d(XX, W1, strides=[1, stride, stride, 1], padding='VALID')
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
Y1r = tf.nn.relu(Y1bn)
Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))

Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
Y2r = tf.nn.relu(Y2bn)
Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))

Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3r = tf.nn.relu(Y3bn)
Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))
Y3_pool = tf.nn.max_pool(Y3, ksize=[1,10,1,1], strides=[1,10,1,1], padding='SAME')

YY = tf.reshape(Y3_pool, shape=[-1, 6 * 1 * M])

Y4l = tf.matmul(YY, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4r = tf.nn.relu(Y4bn)
Y4 = tf.nn.dropout(Y4r, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

cross_entropy_ = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy_)*100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print ('Model built!')

dataX, dataY = load_data(test_file)
dataset = Dataset(dataX, dataY)
__X, __Y = dataset.minibatch(len(dataY))

print ("Data loaded!")

prediction = []

for k in range(k_fold):
    # init
    saver = tf.train.Saver({"W1":W1, "B1":B1,
                            "W2":W2, "B2":B2,
                            "W3":W3, "B3":B3,
                            "W4":W4, "B4":B4,
                            "W5":W5, "B5":B5})
    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init)
    model_path = "cv_model_"+str(k)
    saver.restore(sess, model_path+"/model.ckpt")

    print ('%d-th Model loaded!' % k)

    test_accuracy, pred_ = sess.run((accuracy, Y),
                                    feed_dict={X: __X,
                                               Y_: __Y,
                                               tst: False,
                                               pkeep: 1.0,
                                               pkeep_conv: 1.0})
    print ("  Test set is " + test_file)
    print ("  Test accuracy = %f" % test_accuracy)
    prediction.append(pred_)
    
prediction_ = numpy.asarray(prediction)
prediction_ = prediction_.mean(axis=0).argmax(axis=1)
accuracy = numpy.asarray((prediction_ == __Y.argmax(axis=1)).sum(),
                         dtype="float32") / prediction_.shape[0]
print ("After considering all %d models," % k_fold)
print ("Test accuracy = %f" % accuracy)

l0_acc = 0
l0_cnt = 0
l1_acc = 0
l1_cnt = 0
for i in range(prediction_.shape[0]):
    if __Y.argmax(axis=1)[i] == 0:
        l0_cnt += 1
        if prediction_[i] == 0:
            l0_acc += 1
    else:
        l1_cnt += 1
        if prediction_[i] == 1:
            l1_acc += 1
print ("Label 0 accuracy = %f" % (l0_acc / l0_cnt))
print ("Label 1 accuracy = %f" % (l1_acc / l1_cnt))