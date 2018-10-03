#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:01:28 2018

@author: ericye
"""

import tensorflow as tf
import numpy as np
import pickle
sess = tf.InteractiveSession()
features = pickle.load(open("features.pkl", 'rb'))
labels = pickle.load(open("labels.pkl", 'rb'))

def add_layer(inputs, in_size, out_size,keep_prob=0.5,activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
    biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        outputs = tf.nn.dropout(outputs,keep_prob)  #随机失活
    return outputs
"""
tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。
tf.nn.dropout是TensorFlow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层。
Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
但在测试及验证中：每个神经元都要参加运算，但其输出要乘以概率p。
"""
# holder变量
x = tf.placeholder(tf.float32,[None,1506])
y_ = tf.placeholder(tf.float32,[None,36])
keep_prob = tf.placeholder(tf.float32)     # 概率

"""
tf.placeholder(dtype, shape=None, name=None)
placeholder，占位符，在tensorflow中类似于函数参数，运行时必须传入值
tf.Variable：主要在于一些可训练变量（trainable variables），比如模型的权重（weights，W）或者偏执值（bias）；
声明时，必须提供初始值；
名称的真实含义，在于变量，也即在真实训练时，其值是会改变的，自然事先需要指定初始值；
"""
h1 = add_layer(x,1506,250,keep_prob,tf.nn.sigmoid)
h2 = add_layer(h1,250,200,keep_prob,tf.nn.sigmoid)
h3 = add_layer(h2,200,150,keep_prob,tf.nn.sigmoid)

##输出层
w = tf.Variable(tf.zeros([150,36]))     #300*10
b = tf.Variable(tf.zeros([36]))
y = tf.nn.softmax(tf.matmul(h3,w)+b)

#定义loss,optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
##指定第二个参数为1，则第二维的元素取平均值，即每一行求平均值
train_step  =tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
for i in range(13):
    #correct_prediction = tf.equal(tf.argmax(y[:,0:3],1),tf.argmax(y_[:,0:3],1))       #高维度的
    correct_prediction = tf.equal(tf.argmax(y[:,0+i*3:3+i*3],1),tf.argmax(y_[:,0+i*3:3+i*3],1))       #高维度的

#argmax axis=0竖直方向最大的数所在的下标；axis=1水平方向最大的数所在的下标
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))    #要用reduce_mean  ?
tf.global_variables_initializer().run()
for i in range(200):
    batch_x,batch_y  = features[i:i+100],labels[i:i+100]
 #   print((batch_x.shape))
    train_step.run({x:batch_x,y_:batch_y,keep_prob:0.75})
    if i%10==0:
        train_accuracy = accuracy.eval({x:batch_x,y_:batch_y,keep_prob:1.0})
        print("step %d,train_accuracy %g"%(i,train_accuracy))

###########test
test_x = features[30000:31000]
test_y = labels[30000:31000]
print("Test accuracy:",accuracy.eval({x:test_x,y_:test_y,keep_prob:1.0}))