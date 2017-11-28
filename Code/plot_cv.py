# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:31:28 2017

@author: DrLC
"""

import matplotlib.pyplot as plt
import pickle
import numpy

mean_seg = 1

f = open('plot_cv', 'rb')
_train_acc, _train_loss, _train_it, _test_acc, _test_it = pickle.load(f)
f.close()

ax1 = plt.subplot(111)
ax1.set_title("10-fold CNN Train/Validation Accuracy Curves")
ax1.set_xlabel("iteration")
ax1.set_ylabel("accuracy")

for i in range(len(_train_acc)):
    train_acc = _train_acc[i]
    train_loss = _train_loss[i]
    train_it = _train_it[i]
    test_acc = _test_acc[i]
    test_it = _test_it[i]
    
    new_train_acc = []
    new_test_acc = []
    for i in range(len(train_acc)):
        l = numpy.max([0, i-mean_seg])
        r = numpy.min([len(train_acc), i+mean_seg+1])
        tmp_acc = numpy.mean(train_acc[l:r])
        new_train_acc.append(tmp_acc)
    for i in range(len(test_acc)):
        l = numpy.max([0, i-mean_seg])
        r = numpy.min([len(test_acc), i+mean_seg+1])
        tmp_acc = numpy.mean(test_acc[l:r])
        new_test_acc.append(tmp_acc)

    ax1.plot(train_it,
             new_train_acc,
             color=(1, 0, 0, 0.1),
             linewidth=0.5)
    ax1.plot(test_it,
             new_test_acc,
             color=(0, 1, 0, 0.2),
             linewidth=1)

train_acc = numpy.mean(_train_acc, axis=0)
test_acc = numpy.mean(_test_acc, axis=0)
new_train_acc = []
new_test_acc = []
for i in range(len(train_acc)):
    l = numpy.max([0, i-mean_seg])
    r = numpy.min([len(train_acc), i+mean_seg+1])
    tmp_acc = numpy.mean(train_acc[l:r])
    new_train_acc.append(tmp_acc)
for i in range(len(test_acc)):
    l = numpy.max([0, i-mean_seg])
    r = numpy.min([len(test_acc), i+mean_seg+1])
    tmp_acc = numpy.mean(test_acc[l:r])
    new_test_acc.append(tmp_acc)
ax1.plot(train_it,
         new_train_acc,
         color=(1, 0, 0.1, 0.25),
         linewidth=1,
         label="train acc")
ax1.plot(test_it,
         new_test_acc,
         color=(0, 1, 0.1, 0.5),
         linewidth=2,
         label=("validation acc"))

    
lgd1 = ax1.legend(loc="lower right", fontsize="small")
plt.show()