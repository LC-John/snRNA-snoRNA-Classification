# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 01:12:27 2017

@author: DrLC
"""

import matplotlib.pyplot as plt
import pickle
import numpy

f = open('plot', 'rb')
train_acc, train_loss, train_it, test_acc, test_it = pickle.load(f)
f.close()

new_train_acc = []
new_test_acc = []
mean_seg = 1
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

baseline_acc = [0.5, 0.5]
baseline_it = [0, numpy.max([train_it[-1], test_it[-1]])]

ax1 = plt.subplot(111)
ax1.set_title("CNN on snRNA/snoRNA Classification")
ax1.plot(train_it,
         new_train_acc,
         color=(1, 0, 0, 0.1),
         linewidth=1,
         label="train acc")
ax1.plot(test_it,
         new_test_acc,
         color=(0, 1, 0, 0.5),
         linewidth=2,
         label="test acc")
ax1.set_xlabel("iteration")
ax1.set_ylabel("accuracy")
ax2 = ax1.twinx()
ax2.plot(train_it,
         train_loss,
         color=(0, 0, 1, 0.1),
         linewidth=1,
         label="train loss")
ax2.set_ylabel("loss")

lgd1 = ax1.legend(loc=(0.78,0.50), fontsize="small")
lgd2 = ax2.legend(loc=(0.78,0.43), fontsize="small")
plt.show()