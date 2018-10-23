import tensorflow as tf 
import os 

import tensorflow as tf  
import os  
class ADDA():
    def __init__(self,classes_num):
        self.classes_num = classes_num
        self.s_e = 's_e'
        self.t_e = 't_e'
        self.c = 'c_'
        self.d = 'd_'

    def s_encoder(self,inputs,reuse=False,trainable=True):
        with tf.variable_scope(self.s_e,reuse=reuse):
            conv1 = tf.layers.conv2d(inputs,filters=20,kernel_size=(5,5),activation=tf.nn.relu,trainable=trainable,name='conv1')
            conv1_pooling = tf.layers.max_pooling2d(conv1,(2,2),(2,2),name='pool1')
            conv2 = tf.layers.conv2d(conv1_pooling,filters=50,kernel_size=(5,5),activation=tf.nn.relu,trainable=trainable,name='conv2')
            conv2_pooling = tf.layers.max_pooling2d(conv2,(2,2),(2,2),name='pool2')
            flat = tf.layers.flatten(conv2_pooling,name='flat')
            fc1 = tf.layers.dense(flat,120,activation=tf.nn.relu,trainable=trainable,name='fc1')
            fc2 = tf.layers.dense(fc1,84,activation=tf.nn.tanh,trainable=trainable,name='fc2')
        return fc2

    def t_encoder(self,inputs,reuse=False,trainable=True):
        with tf.variable_scope(self.t_e,reuse=reuse):
            conv1 = tf.layers.conv2d(inputs,filters=20,kernel_size=(5,5),activation=tf.nn.relu,trainable=trainable,name='conv1')
            conv1_pooling = tf.layers.max_pooling2d(conv1,(2,2),(2,2),name='pool1')
            conv2 = tf.layers.conv2d(conv1_pooling,filters=50,kernel_size=(5,5),activation=tf.nn.relu,trainable=trainable,name='conv2')
            conv2_pooling = tf.layers.max_pooling2d(conv2,(2,2),(2,2),name='pool2')
            flat = tf.layers.flatten(conv2_pooling,name='flat')
            fc1 = tf.layers.dense(flat,120,activation=tf.nn.relu,trainable=trainable,name='fc1')
            fc2 = tf.layers.dense(fc1,84,activation=tf.nn.tanh,trainable=trainable,name='fc2')
        return fc2

    def classifier(self,inputs,reuse=False,trainable=True):
        with tf.variable_scope(self.c,reuse=reuse):
            fc = tf.layers.dense(inputs,self.classes_num,activation=None,trainable=trainable,name='fc1')
        return fc
    
    def discriminator(self,inputs,reuse=False,trainable=True):
        with tf.variable_scope(self.d,reuse=reuse):
            fc1 = tf.layers.dense(inputs,128,activation=tf.nn.leaky_relu,trainable=trainable,name='fc1')
            fc2 = tf.layers.dense(fc1,128,activation=tf.nn.leaky_relu,trainable=trainable,name='fc2')
            fc3 = tf.layers.dense(fc2,1,activation=None,trainable=trainable,name='fc3')
            return fc3
    
    def build_classify_loss(self,logits,labels):
        c_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
        c_loss = tf.reduce_mean(c_loss)
        return c_loss
    
    # wgan loss function   
    def build_w_loss(self,disc_s,disc_t):
        d_loss = -tf.reduce_mean(disc_s) + tf.reduce_mean(disc_t)
        g_loss = -tf.reduce_mean(disc_t)
        tf.summary.scalar("g_loss",g_loss)
        tf.summary.scalar('d_loss',d_loss)
        return g_loss,d_loss

    def build_ad_loss(self,disc_s,disc_t):
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t,labels=tf.ones_like(disc_t))
        g_loss = tf.reduce_mean(g_loss)
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_s,labels=tf.ones_like(disc_s)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_t,labels=tf.zeros_like(disc_t)))
        tf.summary.scalar("g_loss",g_loss)
        tf.summary.scalar('d_loss',d_loss)
        return g_loss,d_loss

    def build_ad_loss_v2(self,disc_s,disc_t):
        d_loss = -tf.reduce_mean(tf.log(disc_s+ 1e-12)+tf.log(1-disc_t+1e-12))
        g_loss = -tf.reduce_mean(tf.log(disc_t + 1e-12))
        return g_loss,d_loss

    def eval(self,logits,labels):
        pred = tf.nn.softmax(logits)
        correct_label_predicted = tf.equal(labels,tf.cast(tf.argmax(pred,axis=1),tf.int32))
        predicted_accuracy = tf.reduce_mean(tf.cast(correct_label_predicted,tf.float32))
        return predicted_accuracy

    # used for debug
    # def conv2d(self,inputs,filters,kernel_size,activation,trainable=True,name="conv"):
    #     with tf.variable_scope(name):
    #         w = tf.get_variable("conv",shape=[kernel_size[0],kernel_size[1],inputs.get_shape()[-1],filters],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01),trainable=trainable)
    #         b = tf.get_variable("bias",shape=[filters],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=trainable)
    #         conv = tf.nn.conv2d(inputs,w,(1,1,1,1),padding="VALID")
    #         conv = tf.nn.bias_add(conv,b)
    #         conv = activation(conv)
    #         tf.summary.histogram("conv_w",w)
    #     return conv
    
    # def dense(self,inputs,out_features,activation=None,trainable=True,name="dense"):
    #     with tf.variable_scope(name):
    #         w = tf.get_variable("kernel",shape=[inputs.get_shape()[-1],out_features],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.001),trainable=trainable)
    #         b = tf.get_variable('bias',shape=[out_features],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=trainable)
    #         out = tf.matmul(inputs,w)+b
    #         if activation!=None:
    #             out = activation(out)
    #         tf.summary.histogram("kernel",w)
    #     return out
    
