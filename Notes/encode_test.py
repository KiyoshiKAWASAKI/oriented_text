#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:46:52 2017

@author: xingdaitao
"""
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

isess = tf.InteractiveSession()

shape = (64,64)
np_label = np.array([np.random.randint(2) for i in range(shape[0]*shape[1])]).reshape(shape)

np_label
label = tf.constant(np_label,dtype=tf.int32)
full_label = tf.pad(label, [[1,1],[1,1]], "CONSTANT")

label_flat = tf.reshape(label,[-1,1])

indices = tf.expand_dims(tf.range(shape[0]*shape[1],dtype=tf.int32), -1)


com_label = tf.concat([indices,label_flat],axis=1)

isess.run(com_label)

def update_in(x,full_label=full_label, shape=shape):
    i = tf.div(x[0],shape[0])
    j = tf.mod(x[0],shape[0])
    return tf.cond(tf.less(x[1], 1), 
                   lambda :tf.ones([8,],dtype=tf.int32), 
                   lambda: in_layer(full_label, i+1, j+1))


def in_layer(full_label, i, j):
    indice = [[i-1,j-1],[i-1,j],[i-1,j+1],
              [i,j-1],[i,j+1],
              [i+1,j-1],[i+1,j],[i+1,j+1]]
    values = [full_label[k[0],k[1]] for k in indice]
    return tf.stack(values)

isess.run(tf.map_fn(update_in,com_label))
isess.run(tf.map_fn(update_in,com_label)).shape



## test cross-layer encode
label0 = label
shape1 = [i/2 for i in shape]
np_label1 = np.array([np.random.randint(2) for i in range(shape1[0]*shape1[1])]).reshape(shape1)
label1 = tf.constant(np_label1,dtype=tf.int32)

label_flat1 = tf.reshape(label1,[-1,1])

indices1 = tf.expand_dims(tf.range(shape1[0]*shape1[1],dtype=tf.int32), -1)


com_label1 = tf.concat([indices1,label_flat1],axis=1)

isess.run(com_label1)


def update_cross(x,label=label, shape=shape1):
    i = tf.div(x[0],shape[0])
    j = tf.mod(x[0],shape[0])
    return tf.cond(tf.less(x[1], 1), 
                   lambda :tf.ones([4,],dtype=tf.int32), 
                   lambda :cross_layer(label, i, j))


def cross_layer(label,i,j):
    indice = [[2*i,2*j],[2*i,2*j+1],[2*i+1,2*j],[2*i+1,2*j+1]]
    values = [label[k[0],k[1]] for k in indice]
    values_c = [values[0]*values[1],
                values[1]*values[3],
                values[0]*values[2],
                values[1]*values[3]]
    return tf.stack(values_c)

isess.run(tf.map_fn(update_cross,com_label1))

#%%
isess.run(tf.map_fn(update_cross,com_label1)).shape
#%%
