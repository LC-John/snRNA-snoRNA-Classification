# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:56:45 2017

@author: DrLC
"""

import pickle, gzip
import numpy
import random

def dirty_data(clean_path = "../Dataset/final.pkl.gz",
               dirty_path = "../Dataset/final_tr.pkl.gz", dirty_ratio = 1.,
               final_path = "../Dataset/dirty.pkl.gz"):
    
    with gzip.open(clean_path, 'rb') as f:
        clean = pickle.load(f)
        f.close()
    with gzip.open(dirty_path, 'rb') as f:
        dirty = pickle.load(f)
        f.close()
    
    assert dirty_ratio >= 0. and dirty_ratio <= 1.
    
    dirty_idx = random.sample(range(len(dirty['seq'])),
                              int(len(dirty['seq'])*dirty_ratio))
    data = {}
    data['seq'] = clean['seq']
    data['name'] = clean['name']
    data['label'] = clean["label"]
    for i in dirty_idx:
        data['seq'].append(dirty['seq'][i])
        data['label'].append(2)
        data['name'].append(dirty['name'][i])
    
    with gzip.open(final_path, 'wb') as f:
        pickle.dump(data, f)
        f.close()

def generate_data(paths = ["../Dataset/snRNA.txt",
                           "../Dataset/snRNA_name.txt", 
                           "../Dataset/snoRNA.txt",
                           "../Dataset/snoRNA_name.txt"],
                  final_path = "../Dataset/final.pkl.gz"):
    
    data = []         
    for path in paths:
        with open(path, 'rb') as f:
            data.append(pickle.load(f))
            f.close()
    
    new_data = [[], [], []]
    for i in range(len(data[0])):
        if data[0][i] not in new_data[0]:
            new_data[0].append(data[0][i])
            new_data[1].append(0)
            new_data[2].append(data[1][i])
    for i in range(len(data[2])):
        if data[2][i] not in new_data[0]:
            new_data[0].append(data[2][i])
            new_data[1].append(1)
            new_data[2].append(data[3][i])
            
    with gzip.open(final_path, 'wb') as f:
        pickle.dump({"seq": new_data[0],
                     "label": new_data[1],
                     "name": new_data[2]}, f)
        f.close()
    return new_data[0], new_data[1], new_data[2]

def generate_data_half(paths = ["../Dataset/snRNA.txt",
                                "../Dataset/snRNA_name.txt"],
                       label = 0,
                       final_path = "../Dataset/final_0.pkl.gz"):
    
    data = []         
    for path in paths:
        with open(path, 'rb') as f:
            data.append(pickle.load(f))
            f.close()
    
    new_data = [[], [], []]
    for i in range(len(data[0])):
        if data[0][i] not in new_data[0]:
            new_data[0].append(data[0][i])
            new_data[1].append(label)
            new_data[2].append(data[1][i])
            
    with gzip.open(final_path, 'wb') as f:
        pickle.dump({"seq": new_data[0],
                     "label": new_data[1],
                     "name": new_data[2]}, f)
        f.close()
    return new_data[0], new_data[1], new_data[2]

def load_data(path="../Dataset/final.pkl.gz"):
    
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
        f.close()
        
    return (numpy.asarray(data["seq"]),
            numpy.asarray(data["label"]))

def split_data(X, Y, ratio = [8, 1, 1]):
    
    assert len(X.shape) == 1
    assert len(Y.shape) == 1
    assert X.shape == Y.shape
    assert len(ratio) == 3 or len(ratio) == 2

    X = numpy.copy(X)
    Y = numpy.copy(Y)
    ratio = numpy.array(ratio, copy=True, dtype=numpy.float32)
    if len(ratio) == 3:
        ratio[2] = ratio[0] + ratio[1] + ratio[2]
        ratio[1] = ratio[0] + ratio[1]
        ratio[0] = ratio[0] / ratio[2]
        ratio[1] = ratio[1] / ratio[2]
        ratio[2] = ratio[2] / ratio[2]
    else:
        ratio[1] = ratio[0] + ratio[1]
        ratio[0] = ratio[0] / ratio[1]
        ratio[1] = ratio[1] / ratio[1]

    size = len(Y)
    idx = random.sample(range(size), size)
    tr = [[], []]
    te = [[], []]
    va = [[], []]
    for i in range(int(size*ratio[0])):
        tr[0].append(X[idx[i]])
        tr[1].append(Y[idx[i]])
    for i in range(int(size*ratio[0]), int(size*ratio[1])):
        te[0].append(X[idx[i]])
        te[1].append(Y[idx[i]])
    if len(ratio) == 3:
        for i in range(int(size*ratio[1]), int(size*ratio[2])):
            va[0].append(X[idx[i]])
            va[1].append(Y[idx[i]])
    tr[0] = numpy.asarray(tr[0])
    tr[1] = numpy.asarray(tr[1])
    te[0] = numpy.asarray(te[0])
    te[1] = numpy.asarray(te[1])
    if len(ratio) == 3:
        va[0] = numpy.asarray(va[0])
        va[1] = numpy.asarray(va[1])
        return (tr, te, va)
    else:
        return (tr, te)
    
def cv_split(X, Y, k):
    
    assert len(X.shape) == 1
    assert len(Y.shape) == 1
    assert X.shape == Y.shape

    X = numpy.copy(X)
    Y = numpy.copy(Y)
    size = len(Y)
    idx = random.sample(range(size), size)
    
    sets = []
    for i in range(k):
        sets.append([[], []])
        for j in range(int(size/k*i), int(size/k*(i+1))):
            sets[-1][0].append(X[idx[j]])
            sets[-1][1].append(Y[idx[j]])
        sets[-1][0] = numpy.asarray(sets[-1][0])
        sets[-1][1] = numpy.asarray(sets[-1][1])
    
    return sets
        
def cv_generate(sets, k):
    
    assert len(sets) > k
    assert k >= 0
    
    X, Y, cvX, cvY = (None, None, None, None)
    for i in range(len(sets)):
        if i == k:
            cvX = numpy.copy(sets[i][0])
            cvY = numpy.copy(sets[i][1])
        elif X is None:
            X = numpy.copy(sets[i][0])
            Y = numpy.copy(sets[i][1])
        else:
            X = numpy.copy(numpy.concatenate([X, sets[i][0]]))
            Y = numpy.copy(numpy.concatenate([Y, sets[i][1]]))
        
    return (X, Y), (cvX, cvY)
    
def stat_data(X, Y):
    
    maxlen = 0
    for i in range(len(X)):
        if maxlen < len(X[i]):
            maxlen = len(X[i])
            
    seq_len = [0 for i in range(maxlen+1)]
    for i in X:
        seq_len[len(i)] += 1
    
    label_num = [0, 0]
    for i in Y:
        label_num[i] += 1

    return {"maxlength": maxlen,
            "length": seq_len,
            "label": label_num}

class Dataset():
    
    def __init__(self, X, Y, seqlen=60, rand_seed=1234, 
                 embedding={"A": [0, 0, 0, 1],
                            "C": [0, 0, 1, 0],
                            "G": [0, 1, 0, 0],
                            "T": [1, 0, 0, 0]},
                 padding=[0, 0, 0, 0]):
        
        _X = numpy.copy(X)
        _Y = numpy.copy(Y)
        em_len = len(padding)
        
        assert len(_X.shape) == 1
        assert len(_Y.shape) == 1
        assert _X.shape == _Y.shape
        assert type(embedding) is dict
        for i in embedding.keys():
            assert len(embedding[i]) == em_len
        
        self.X = _X
        self.Y = _Y
        self.seqlen = seqlen
        self.emlen = em_len
        self.__available = random.sample(range(self.Y.shape[0]),
                                         self.Y.shape[0])
        self.__embedding = embedding
        self.__padding = padding
        random.seed(rand_seed)
    
    def minibatch(self, batchsize):
        
        if len(self.__available) < batchsize:
            # print ("reload")
            self.__available = random.sample(range(self.Y.shape[0]),
                                             self.Y.shape[0])
        idx = self.__available[:batchsize]
        self.__available = self.__available[batchsize:]
        
        _X = [self.X[i] for i in idx]
        _Y = [self.Y[i] for i in idx]
        X = []
        Y = []
        for seq in _X:
            X.append([])
            for n in seq:
                X[-1].append(self.__embedding[n])
            for i in range(len(seq), self.seqlen):
                X[-1].append(self.__padding)
        for label in _Y:
            Y.append([0, 0])
            Y[-1][label] = 1
            
        X = numpy.asarray(X, dtype='float32')
        Y = numpy.asarray(Y, dtype='float32')
        
        assert len(Y) == batchsize
        return (X, Y)
        
    def dirty_minibatch(self, batchsize):
        
        if len(self.__available) < batchsize:
            # print ("reload")
            self.__available = random.sample(range(self.Y.shape[0]),
                                             self.Y.shape[0])
        idx = self.__available[:batchsize]
        self.__available = self.__available[batchsize:]
        
        _X = [self.X[i] for i in idx]
        _Y = [self.Y[i] for i in idx]
        X = []
        Y = []
        for seq in _X:
            X.append([])
            for n in seq:
                X[-1].append(self.__embedding[n])
            for i in range(len(seq), self.seqlen):
                X[-1].append(self.__padding)
        for label in _Y:
            Y.append([0, 0, 0])
            Y[-1][label] = 1
            
        X = numpy.asarray(X, dtype='float32')
        Y = numpy.asarray(Y, dtype='float32')
        
        assert len(Y) == batchsize
        return (X, Y)
        

    
if __name__ == "__main__":
    
    '''    
    generate_data(paths = ["../Dataset/snRNA.txt",
                           "../Dataset/snRNA_name.txt", 
                           "../Dataset/snoRNA.txt",
                           "../Dataset/snoRNA_name.txt"],
                  final_path = "../Dataset/final.pkl.gz")
    generate_data(paths = ["../Dataset/snRNA_m.txt",
                           "../Dataset/snRNA_m_name.txt", 
                           "../Dataset/snoRNA_m.txt",
                           "../Dataset/snoRNA_m_name.txt"],
                  final_path = "../Dataset/final_m.pkl.gz")
    '''
    
    generate_data_half(paths = ["../Dataset/trRNA.txt",
                                "../Dataset/trRNA_name.txt"],
                       label=2,
                       final_path = "../Dataset/final_tr.pkl.gz")
    generate_data_half(paths = ["../Dataset/trRNA_m.txt",
                                "../Dataset/trRNA_m_name.txt"],
                       label=2,
                       final_path = "../Dataset/final_m_tr.pkl.gz")

    
    X, Y = load_data()
    
    stat = stat_data(X, Y)
    #(X, Y), (testX, testY), (validX, validY) = split_data(X, Y, [8, 1, 1])
    (trainX, trainY), (testX, testY) = split_data(X, Y, [9, 1])
    print (trainX.shape)
    print (trainY.shape)
    print (testX.shape)
    print (testY.shape)
    
    trainset = Dataset(trainX, trainY)
    testset = Dataset(testX, testY)
    #validset = Dataset(validX, validY)
    
    sets = cv_split(X, Y, 10)
    for i in sets:
        print (i[0].shape, i[1].shape)
    
    dirty_data()
    dX, dY = load_data("../Dataset/dirty.pkl.gz")
    dirtyset = Dataset(dX, dY)