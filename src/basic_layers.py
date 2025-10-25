#!/usr/bin/env python
import tensorflow as tf

def build_resnet_block(input_res, channelsout, is_training, data_format,
                       do_batch_norm=False, name=None):
    inputs = tf.identity(input_res)
    out_res_1 = general_conv2d(input_res,
                               name=name+'_res1',
                               channelsout=channelsout,
                               strides=1,
                               do_batch_norm=do_batch_norm,
                               is_training=is_training,
                               data_format=data_format)
    out_res_2 = general_conv2d(out_res_1,
                               name=name+'_res2',
                               channelsout=channelsout,
                               strides=1,
                               do_batch_norm=do_batch_norm,
                               is_training=is_training,
                               data_format=data_format)
    return out_res_2 + inputs

def general_conv2d(conv, name=None, channelsout=64, ksize=3, strides=2, init_factor=0.1,
                   padding='SAME', do_batch_norm=False, activation=tf.nn.relu,
                   is_training=True, data_format=None):
    
    conv = tf.compat.v1.layers.conv2d(conv,
                            channelsout,
                            ksize,
                            strides=strides,
                            padding=padding,
                            activation=activation,
                            kernel_initializer=tf.compat.v1.variance_scaling_initializer(scale=init_factor),
                            bias_initializer=tf.compat.v1.constant_initializer(0.0),
                            data_format=data_format)

    if do_batch_norm:
        conv = tf.compat.v1.layers.batch_normalization(conv,
                                             axis=1 if data_format=='channels_first' else -1,
                                             epsilon=1e-5,
                                             gamma_initializer=tf.compat.v1.constant_initializer([0.01]),
                                             name=name+'_bn',
                                             training=is_training)
    return conv

"""
Upsample a tensor by a factor of 2 with fixed padding and then do normal conv2d on it. 
Similar operation to a transposed convolution, but avoids checkerboard artifacts.
"""
def upsample_conv2d(conv, name=None, channelsout=64, ksize=3, init_factor=0.1,
                    do_batch_norm=False, is_training=True, data_format=None):
    if data_format == 'channels_first':
        conv = tf.transpose(conv, [0,2,3,1])
        shape = tf.shape(conv)
        conv = tf.image.resize(conv, size=[shape[1]*2, shape[2]*2],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        pad = (ksize - 1) // 2
        paddings = tf.constant([[0, 0],
                                [pad, pad],
                                [pad, pad],
                                [0, 0]], dtype=tf.int32)
        
        conv = tf.pad(conv,
                      paddings = paddings,
                      mode = 'REFLECT')
        
        conv = tf.transpose(conv, [0,3,1,2])
    else:
        shape = tf.shape(conv)
        # conv = tf.image.resize_images(conv, size=[shape[1]*2, shape[2]*2],
        conv = tf.image.resize(conv, size=[shape[1]*2, shape[2]*2],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        pad = (ksize - 1) // 2
        paddings = tf.constant([[0, 0],
                                [pad, pad],
                                [pad, pad],
                                [0, 0]], dtype=tf.int32)
        
        conv = tf.pad(conv,
                paddings = paddings,
                mode = 'REFLECT')

    conv = general_conv2d(conv, name=name, channelsout=channelsout, ksize=ksize, strides=1,
                          do_batch_norm=do_batch_norm, padding='VALID', init_factor=init_factor,
                          is_training=is_training, data_format=data_format)
    
    return conv

def predict_flow(conv, name=None, channelsout=2, ksize=1, strides=1,
                 padding='SAME', init_factor=0.1,
                 is_training=True, data_format=None):
    conv = general_conv2d(conv,
                          channelsout=channelsout,
                          ksize=ksize,
                          strides=strides,
                          init_factor=init_factor,
                          padding=padding,
                          activation=tf.tanh,
                          is_training=is_training,
                          data_format=data_format,
                          name=name)
    return conv